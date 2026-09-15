# onsram/run_onsram.py
"""
OnSRAM-Static entry point -- Phase B+C (the algorithm port: FoM scoring,
greedy whole-interval pinning, BFS-DFS scheduling) plus Phase D (wiring
that decision into real SCALE-Sim via OnSRAM's own
onsram_helpers/scale_sim_runner.py, replacing OnSRAM's original
closed-form latency estimator entirely with real, engine-measured cycle
counts and DRAM byte traffic). See onsram/docs/onsram_integration_plan.md
for the full roadmap.

COSMA and OnSRAM are two separate jobs that happen to sit on the same
SCALE-Sim engine, not one depending on the other's code: graph parsing
and physical-consistency checking reuse cosma/helpers/graph_builder.py
and spm_allocator.py directly, since those are generic, algorithm-
agnostic infrastructure (plain dicts in, plain dicts out, no assumption
about how a plan was produced) already shared the same way by COSMA's
own run_paper_baselines.py. Everything else is OnSRAM's own: the WHERE-
placement step (onsram_helpers/placement.py -- Best-Fit-Decreasing,
OnSRAM's own algorithm, not a borrowed one, see that module's docstring
for why) and Phase D's entire SCALE-Sim-driving logic (a full,
self-contained duplicate living in onsram_helpers/scale_sim_runner.py +
topology.py + resident_buffers.py) -- a change to COSMA's own
baseline.py/topology_builder.py/cosma_resident_buffers.py/
tflite_arena_allocator.py can never change OnSRAM's numbers, and vice
versa.

Pipeline: cosma/helpers/graph_builder.load_graph() -> onsram_helpers'
FoM scoring -> BFS-DFS hybrid scheduling -> liveness analysis -> greedy
whole-interval pinning (with Overwrite Optimization) -> OnSRAM's own
resident_action/spm_plan dict shapes (the latter via
onsram_helpers.placement.place_tensors()) [Phase C, run_onsram()]
-> a real SCALE-Sim simulation pass via
onsram_helpers.scale_sim_runner.run_onsram_aware(), compared against a
plain run_baseline() pass for DRAM traffic and cycle-count savings
[Phase D, run_onsram_scale_sim()]. Both take any model.json path and any
budget -- neither is model-specific.

Usage:
  python3 run_onsram.py                                   # MobileNet @ 2MB (default)
  python3 run_onsram.py --model densenet --spm-mb 1 2 4    # sweep budgets on one model
  python3 run_onsram.py --model densenet --model MobileNet --spm-mb 2  # sweep models
  python3 run_onsram.py --model /path/to/model.json --spm-mb 2
  python3 run_onsram.py --plot                             # also save an SPM-occupancy PNG per run
  python3 run_onsram.py --out-csv results.csv              # also save a cross-combination summary CSV
  python3 run_onsram.py --no-logs                          # skip per-combination log files, terminal only
  python3 run_onsram.py --no-scale-sim                     # Phase C only, skip the slow real SCALE-Sim pass

Phase D is real, engine-driven simulation, not a closed-form estimate --
expect it to be genuinely slow: two full SCALE-Sim passes (a plain
baseline plus an OnSRAM-aware pass) over every conv-like layer, timed
directly at ~3 minutes total for MobileNet's 30 layers alone. A model
with many more conv-like layers (e.g. DenseNet's ~156) will take
correspondingly longer -- a heartbeat line prints to stderr every 20s
during these passes so a long wait doesn't look like a hang (see
_start_heartbeat(), duplicated from cosma/run_experiments.py's own
identical mechanism, needed there for the exact same reason). Pass
--no-scale-sim for a fast, decision-only pass when you don't need the
DRAM/cycle numbers.

Logging: by default, every (model, budget) combination's full verbose
report (pin-decision counts, check_physical_validity()'s [OK] lines, the
MobileNet regression line when applicable, print_detailed_report()'s
schedule/FoM-list/pin-duration sections, and print_dram_savings()'s real
SCALE-Sim numbers) is saved to its own file under onsram/logs/, named
`<model>_<budget>MB.log` -- no timestamp, so re-running the same
combination overwrites its previous log rather than accumulating one
file per run. Same convention as cosma/run_experiments.py's own
_run_and_log()/_log_file_name(). The terminal instead gets one compact
progress line and one compact result line per combination (now including
DRAM-traffic-reduction% and speedup once Phase D runs); pass --no-logs to
send the full report straight to the terminal instead (useful for a
single one-off run).

Validation is split in two:
  - check_physical_validity(): holds for ANY model/budget -- a live
    SpmAllocator replay confirms the produced resident_action/spm_plan has
    no capacity violation or address collision (onsram_helpers.placement.
    place_tensors() already runs this same check internally before
    returning, so this confirms it a second, independent time), and no
    oversized tensor got pinned.
  - validate_mobilenet_regression(): MobileNet@2MB-specific, hand-derived
    facts about that ONE fixture (30 COSMA-tracked tensors, identity
    schedule order on its linear chain) -- not general algorithm
    invariants, so only run for that exact model/budget combination.

NOT validated against /home/george/trim/spm_management/output.txt's exact
reported MobileNet-at-2MB pin ratio (29-30/31) -- that number is only
reachable via a real bug in the reference's get_live_timesteps() (an
exclusive-end live range that undercounts a tensor's true footprint at
its last-use timestep; see onsram_helpers/pinning.py's module docstring
for the full empirical trace). This port uses the physically-correct
inclusive range instead (matching the OnSRAM paper's own Sec 4.2
definition of pinning), which a real SpmAllocator replay confirms is
placeable without capacity violations or address collisions -- something
the reference's own numbers, if fed through COSMA's placement
infrastructure, are NOT.
"""
import argparse
import contextlib
import csv
import os
import sys
import threading
import time
import traceback
from typing import Tuple

_ONSRAM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_ONSRAM_DIR)
_COSMA_DIR = os.path.join(_REPO_ROOT, 'cosma')
if _COSMA_DIR not in sys.path:
    sys.path.insert(0, _COSMA_DIR)
# onsram_helpers.scale_sim_runner imports scalesim directly (from
# scalesim.scale_config import scale_config) -- that package lives at the
# repo root, not under cosma/, so it needs its own sys.path entry too.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from helpers import graph_builder                    # COSMA's, unmodified
from helpers import model_resolver                    # COSMA's, unmodified
from helpers.spm_allocator import SpmAllocator        # COSMA's, unmodified

from onsram_helpers import fom, scheduling, pinning, visualize, scale_sim_runner

DEFAULT_MODEL = 'MobileNet'
DEFAULT_SPM_MB = 2.0
DEFAULT_PLOT_DIR = os.path.join(_ONSRAM_DIR, 'spm_plots')
DEFAULT_LOGS_DIR = os.path.join(_ONSRAM_DIR, 'logs')
DEFAULT_CONFIG = os.path.join(os.path.dirname(_COSMA_DIR), 'configs', 'scale.cfg')
CONV_LIKE_OPS = ('CONV2D', 'DEPTHWISE_CONV2D')


def _default_bandwidth_words_per_cycle(config_path: str) -> float:
    """
    Duplicated (not imported) from cosma/run_cosma.py's own private
    helper of the same name -- same isolation rationale as everywhere
    else in this port (see onsram_helpers/pinning.py's own module
    docstring): a fully separate, additive file re-derives what it needs
    from a sibling module's own logic rather than reaching into its
    private (leading-underscore) internals, which aren't part of any
    stable contract. SCALE-Sim's scale.cfg here runs in CALC (estimate)
    bandwidth mode, which doesn't expose one unified DRAM bandwidth
    number -- it estimates per-buffer backing bandwidth from the array
    width instead (see single_layer_sim.run()'s `ofmap_backing_bw =
    arr_col` default). We reuse that same array-width estimate as the
    system DRAM bandwidth for converting extra_dram_bytes into cycles.
    """
    from scalesim.scale_config import scale_config
    config = scale_config()
    config.read_conf_file(config_path)
    if config.use_user_dram_bandwidth():
        return float(config.get_bandwidths_as_list()[0])
    _, arr_col = config.get_array_dims()
    return float(arr_col)


_HEARTBEAT_INTERVAL_SEC = 20


def _start_heartbeat(label: str) -> threading.Event:
    """
    Prints a short "still running" line to stderr every
    _HEARTBEAT_INTERVAL_SEC seconds for as long as the returned Event
    stays unset -- same liveness-signal pattern as
    cosma/run_experiments.py's / cosma/run_paper_baselines.py's own
    _start_heartbeat(), duplicated (not imported) per this port's usual
    isolation rationale (see onsram_helpers/pinning.py's module
    docstring). Needed here for the same reason it's needed there: a
    real SCALE-Sim pass over a model with many conv-like layers can run
    for minutes with nothing else printed in between (confirmed directly
    -- MobileNet's 30 layers alone took ~3 minutes for one baseline +
    one OnSRAM-aware pass; a model with an order of magnitude more
    conv-like layers, like DenseNet, will take correspondingly longer).
    Caller must call .set() on the returned Event once the covered step
    finishes.
    """
    stop = threading.Event()
    start = time.monotonic()

    def _beat():
        while not stop.wait(_HEARTBEAT_INTERVAL_SEC):
            elapsed = time.monotonic() - start
            print(f"[heartbeat] still running: {label} elapsed={elapsed:.0f}s", file=sys.stderr)

    threading.Thread(target=_beat, daemon=True).start()
    return stop


def resolve_model_arg(model_arg: str) -> str:
    """
    Accepts a full path to a model.json, a .tflite path (resolved via
    COSMA's own model_resolver.resolve_model_json(), reused unmodified),
    or a bare model name that resolves to cosma/_exported/<name>/model.json
    -- the export layout cosma/helpers/model_resolver.py itself produces.
    """
    if model_arg.endswith('.json'):
        return model_arg
    if model_arg.endswith('.tflite'):
        return model_resolver.resolve_model_json(model_arg)
    candidate = os.path.join(_COSMA_DIR, '_exported', model_arg, 'model.json')
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"could not resolve model '{model_arg}' to a model.json -- pass a full "
        f"path, a .tflite path, or a name under {os.path.join(_COSMA_DIR, '_exported')}")


def run_onsram(model_json_path: str, memory_budget_bytes: int,
               verbose: bool = True) -> dict:
    nodes, tensors = graph_builder.load_graph(model_json_path)
    layer_meta = fom.load_layer_meta(model_json_path)

    node_flops = fom.estimate_node_flops(layer_meta)
    reuse_meta = fom.build_reuse_meta(layer_meta)
    node_type = fom.classify_all_nodes(nodes, layer_meta, node_flops)

    schedule_order = scheduling.schedule_bfs_dfs_hybrid(nodes, tensors, node_type)
    schedule = list(enumerate(schedule_order))  # [(t, layer_id), ...], COSMA convention

    produced_at_ts, last_used_at_ts, reuse_count = scheduling.analyze_liveness(
        nodes, tensors, schedule_order)

    tensor_fom = fom.calculate_fom(nodes, tensors, schedule, reuse_meta, node_flops,
                                    produced_at_ts, last_used_at_ts)

    pinned, reclaimed_source_ids = pinning.decide_pinning(
        tensors, produced_at_ts, last_used_at_ts, tensor_fom,
        schedule_len=len(schedule), capacity_bytes=memory_budget_bytes)

    resident_action = pinning.build_resident_action(
        tensors, pinned, produced_at_ts, last_used_at_ts, reclaimed_source_ids)
    spm_plan = pinning.build_spm_plan(tensors, resident_action, memory_budget_bytes)

    pinned_count = sum(pinned.values())
    total = len(tensors)
    if verbose:
        print(f"OnSRAM-Static @ {memory_budget_bytes / 1024 / 1024:.2f} MB: "
              f"{pinned_count}/{total} COSMA-tracked tensors pinned "
              f"({100 * pinned_count / total:.1f}%)")

    return {
        'nodes': nodes, 'tensors': tensors, 'schedule': schedule,
        'pinned': pinned, 'resident_action': resident_action,
        'spm_plan': spm_plan, 'tensor_fom': tensor_fom,
        'schedule_order': schedule_order,
        'produced_at_ts': produced_at_ts, 'last_used_at_ts': last_used_at_ts,
        'reclaimed_source_ids': reclaimed_source_ids,
    }


def run_onsram_scale_sim(result: dict, model_json_path: str, config_path: str,
                          memory_budget_bytes: int, layer_stats: dict,
                          bandwidth_bytes_per_cycle: float = None,
                          verbose: bool = False) -> dict:
    """
    Phase D: drives OnSRAM's own resident_action/spm_plan (from
    run_onsram()) through OnSRAM's own, self-contained SCALE-Sim wiring
    (onsram_helpers.scale_sim_runner.run_onsram_aware() -- a duplicate of
    COSMA's baseline.run_cosma_aware(), not an import of it; see
    scale_sim_runner.py's module docstring for why) instead of trusting
    the decision layer's own bookkeeping -- this is what finally replaces
    OnSRAM's original closed-form latency estimator (dropped entirely per
    onsram/docs/onsram_integration_plan.md's Phase D) with real,
    engine-measured cycle counts and DRAM byte traffic.

    layer_stats: scale_sim_runner.run_baseline()'s output for this model --
        the plain, no-management simulation. Schedule- and budget-independent
        (baseline never reorders or caches anything), so callers sweeping
        several budgets for the same model should compute this once and
        pass it in (see __main__ below) rather than re-running SCALE-Sim
        per budget -- same efficiency pattern COSMA's own run_cosma.py/
        run_experiments.py use for their own layer_stats.

    The accounting below is a deliberately simplified duplicate of
    run_cosma.py's own idealized-vs-real accounting block (same
    isolation rationale as elsewhere in this port -- see
    onsram_helpers/pinning.py's module docstring): OnSRAM never spills or
    retrieves a tensor (whole-lifetime pinning is all-or-nothing, see
    onsram/docs/onsram_integration_plan.md sec 2), so COSMA's own
    idealized_spill_bytes/idealized_retrieve_bytes/real_retrieve_bytes
    terms (which exist only to cost a 'S'/'R' action) are structurally
    always zero here and are dropped rather than carried over unused.
    What's left -- and what actually drives every byte saved in this
    port -- is the same "residency credit" concept COSMA's own accounting
    uses: bytes SCALE-Sim's engine confirms are avoidable simply by
    keeping an activation on-chip across layers (ifmap credit, from
    resident_action 'P' entries) plus every layer's own output never
    needing an immediate DRAM round-trip at creation (ofmap credit --
    unconditional whenever resident_action is not None, independent of
    any specific pinning choice; see resident_buffers.py's docstring).
    Reported separately, not just as one combined total, so the log
    doesn't conflate "saved because OnSRAM chose to pin this" with "saved
    because run_onsram_aware() gives every layer's ofmap this for free
    regardless."
    """
    if bandwidth_bytes_per_cycle is None:
        bandwidth_bytes_per_cycle = _default_bandwidth_words_per_cycle(config_path)

    nodes = result['nodes']
    tensors = result['tensors']
    schedule = result['schedule']
    resident_action = result['resident_action']

    heartbeat_stop = _start_heartbeat(
        f"OnSRAM-aware SCALE-Sim pass for {_model_name(model_json_path)} "
        f"@ {memory_budget_bytes / 1024 / 1024:.2f} MB")
    try:
        onsram_stats = scale_sim_runner.run_onsram_aware(
            model_json_path, config_path, resident_action,
            spm_plan=result['spm_plan'], tensors=tensors,
            memory_budget_bytes=memory_budget_bytes, schedule=schedule, verbose=verbose)
    finally:
        heartbeat_stop.set()

    baseline_total_cycles = 0
    onsram_total_cycles = 0
    baseline_dram_bytes = 0
    onsram_dram_bytes = 0
    total_ifmap_residency_credit_bytes = 0
    total_ofmap_residency_credit_bytes = 0
    layer_bound_breakdown = []

    for t, lid in schedule:
        base_s = layer_stats[lid]
        onsram_s = onsram_stats[lid]
        node = nodes[lid]

        total_ifmap_residency_credit_bytes += (
            base_s['ifmap_dram_bytes'] - onsram_s['ifmap_dram_bytes'])
        total_ofmap_residency_credit_bytes += (
            base_s['ofmap_dram_bytes'] - onsram_s['ofmap_dram_bytes'])

        baseline_compulsory = (base_s['ifmap_dram_bytes'] + base_s['ofmap_dram_bytes']
                                + base_s['filter_dram_bytes'])
        onsram_compulsory = (onsram_s['ifmap_dram_bytes'] + onsram_s['ofmap_dram_bytes']
                              + onsram_s['filter_dram_bytes'])

        baseline_mem_cycles = baseline_compulsory / bandwidth_bytes_per_cycle
        onsram_mem_cycles = onsram_compulsory / bandwidth_bytes_per_cycle
        layer_bound_breakdown.append({
            't': t, 'layer_id': lid, 'op': node.op,
            'baseline_compute_cycles': base_s['compute_cycles'],
            'baseline_mem_cycles': baseline_mem_cycles,
            'baseline_bound': 'memory' if baseline_mem_cycles > base_s['compute_cycles'] else 'compute',
            'onsram_compute_cycles': onsram_s['compute_cycles'],
            'onsram_mem_cycles': onsram_mem_cycles,
            'onsram_bound': 'memory' if onsram_mem_cycles > onsram_s['compute_cycles'] else 'compute',
        })

        baseline_total_cycles += max(base_s['compute_cycles'], baseline_mem_cycles)
        onsram_total_cycles += max(onsram_s['compute_cycles'], onsram_mem_cycles)
        baseline_dram_bytes += baseline_compulsory
        onsram_dram_bytes += onsram_compulsory

    dram_traffic_reduction_pct = (
        100.0 * (baseline_dram_bytes - onsram_dram_bytes) / baseline_dram_bytes
        if baseline_dram_bytes > 0 else 0.0)
    speedup = (baseline_total_cycles / onsram_total_cycles
               if onsram_total_cycles > 0 else float('inf'))

    return {
        'onsram_stats': onsram_stats,
        'baseline_total_cycles': baseline_total_cycles,
        'onsram_total_cycles': onsram_total_cycles,
        'baseline_dram_bytes': baseline_dram_bytes,
        'onsram_dram_bytes': onsram_dram_bytes,
        'dram_traffic_reduction_pct': dram_traffic_reduction_pct,
        'speedup': speedup,
        'total_ifmap_residency_credit_bytes': total_ifmap_residency_credit_bytes,
        'total_ofmap_residency_credit_bytes': total_ofmap_residency_credit_bytes,
        'bandwidth_bytes_per_cycle': bandwidth_bytes_per_cycle,
        'layer_bound_breakdown': layer_bound_breakdown,
    }


def print_dram_savings(scale_sim_result: dict) -> None:
    """The actual "how much DRAM traffic did we save" report for the log --
    real, engine-measured numbers from run_onsram_scale_sim(), not the
    decision layer's own estimate."""
    r = scale_sim_result
    print("\n--- REAL SCALE-SIM RESULTS (Phase D: scale_sim_runner.run_onsram_aware()) ---")
    print(f"  baseline DRAM bytes (no SPM mgmt):      {r['baseline_dram_bytes']:>12,d}")
    print(f"  OnSRAM-aware DRAM bytes:                 {r['onsram_dram_bytes']:>12,d}")
    print(f"  DRAM traffic reduction:                  {r['dram_traffic_reduction_pct']:>11.2f}%")
    print(f"    of which, ifmap residency credit:      {r['total_ifmap_residency_credit_bytes']:>12,d} bytes "
          f"(pinned tensors read from SPM instead of DRAM)")
    print(f"    of which, ofmap residency credit:      {r['total_ofmap_residency_credit_bytes']:>12,d} bytes "
          f"(every layer's own output never round-trips DRAM at creation -- Eq.3, "
          f"unconditional in this mode, not specific to OnSRAM's pinning choices)")
    print(f"  baseline total cycles:                   {r['baseline_total_cycles']:>12,.0f}")
    print(f"  OnSRAM-aware total cycles:                {r['onsram_total_cycles']:>12,.0f}")
    print(f"  speedup (baseline_cycles / onsram_cycles): {r['speedup']:.4f}x")
    memory_bound_baseline = sum(1 for x in r['layer_bound_breakdown'] if x['baseline_bound'] == 'memory')
    memory_bound_onsram = sum(1 for x in r['layer_bound_breakdown'] if x['onsram_bound'] == 'memory')
    print(f"  memory-bound layers: baseline={memory_bound_baseline}, "
          f"onsram-aware={memory_bound_onsram} (of {len(r['layer_bound_breakdown'])} total)")


def check_physical_validity(result: dict, memory_budget_bytes: int,
                             verbose: bool = True) -> dict:
    """
    Checks that hold for ANY model/budget combination -- not specific to
    one fixture. A real SpmAllocator replay is the load-bearing check:
    it independently re-verifies every Create/Preserve transition in
    resident_action/spm_plan is physically realizable (no capacity
    violation, no address collision) -- onsram_helpers.placement.
    place_tensors() already runs this same check internally before
    returning, so this confirms the result a second, independent time.

    Returns a small stats dict (pinned_count, total_tensors, peak_bytes,
    oversized_count) so callers (e.g. the CSV summary row / log header)
    don't need to recompute the same numbers a second time.
    """
    tensors = result['tensors']
    pinned = result['pinned']

    oversized_tensor_ids = {tid for tid, t in tensors.items() if t.size_bytes > memory_budget_bytes}
    for tid in oversized_tensor_ids:
        assert not pinned[tid], (
            f"tensor {tid} ({tensors[tid].size_bytes / 1024 / 1024:.4f} MB) exceeds the "
            f"{memory_budget_bytes / 1024 / 1024:.2f} MB budget but was pinned anyway")

    allocator = SpmAllocator(tensors, result['spm_plan'], result['resident_action'],
                              memory_budget_bytes)
    allocator.replay_all()  # raises SpmAllocationError on any physical inconsistency
    peak_bytes = allocator.peak_occupied_bytes()
    assert peak_bytes <= memory_budget_bytes, (
        f"peak SPM usage {peak_bytes} bytes exceeds budget {memory_budget_bytes} bytes")

    pinned_count = sum(pinned.values())
    if verbose:
        print(f"  [OK] {pinned_count}/{len(tensors)} pinned, independently replayed with no "
              f"physical inconsistency; peak {peak_bytes / 1024 / 1024:.4f} MB / "
              f"{memory_budget_bytes / 1024 / 1024:.2f} MB budget "
              f"(oversized excluded: {sorted(oversized_tensor_ids)})")

    return {
        'pinned_count': pinned_count, 'total_tensors': len(tensors),
        'peak_bytes': peak_bytes, 'oversized_count': len(oversized_tensor_ids),
    }


def print_detailed_report(result: dict, memory_budget_bytes: int) -> None:
    """
    The actual per-run detail a log file is for: the chosen execution
    schedule, every tensor's FoM (sorted, descending -- same spirit as
    the reference's own "[FoM] Top tensors by merit" printout, but for
    every tensor, not just the top 10, since this goes to a file rather
    than the terminal), and a breakdown of how many pinned tensors were
    actually kept resident across more than one timestep (a genuine
    cross-layer cache hit) vs. pinned for exactly one timestep (produced
    and consumed within the same instant -- still a correct decision,
    just not really "caching" anything across layers).

    Timestep counts here come directly from resident_action's own entries
    (not re-derived from produced_at_ts/last_used_at_ts), since a
    reclaimed-source tensor's materialized residency is one timestep
    shorter than its raw liveness -- see pinning.py's module docstring.
    """
    nodes = result['nodes']
    tensors = result['tensors']
    pinned = result['pinned']
    tensor_fom = result['tensor_fom']
    produced_at_ts = result['produced_at_ts']
    last_used_at_ts = result['last_used_at_ts']
    reclaimed_source_ids = result['reclaimed_source_ids']
    resident_action = result['resident_action']
    schedule_order = result['schedule_order']

    print("\n--- SCHEDULE (BFS-DFS hybrid execution order) ---")
    for t, layer_id in enumerate(schedule_order):
        print(f"  t={t:<4d} layer={layer_id:<5d} op={nodes[layer_id].op}")

    print("\n--- FOM LIST (all tensors, sorted by FoM descending) ---")
    print(f"  {'tensor':>7} {'size_KB':>10} {'produced':>8} {'last_used':>9} "
          f"{'fom':>10}  pinned  reclaimed_src")
    for tid in sorted(tensors, key=lambda t: -tensor_fom[t]):
        print(f"  {tid:>7} {tensors[tid].size_bytes / 1024:>10.2f} "
              f"{produced_at_ts[tid]:>8} {last_used_at_ts[tid]:>9} "
              f"{tensor_fom[tid]:>10.4f}  {str(pinned[tid]):<6}  {tid in reclaimed_source_ids}")

    timesteps_resident = {}
    for (tid, t), action in resident_action.items():
        if action in ('C', 'P'):
            timesteps_resident[tid] = timesteps_resident.get(tid, 0) + 1

    pinned_ids = [tid for tid, is_pinned in pinned.items() if is_pinned]
    multi_timestep = [tid for tid in pinned_ids if timesteps_resident.get(tid, 0) > 1]
    single_timestep = [tid for tid in pinned_ids if timesteps_resident.get(tid, 0) == 1]
    oversized_count = sum(1 for t in tensors.values() if t.size_bytes > memory_budget_bytes)

    print("\n--- PIN DURATION SUMMARY ---")
    print(f"  total tensors:                         {len(tensors)}")
    print(f"  pinned:                                 {len(pinned_ids)}")
    print(f"  pinned for MORE than one timestep:      {len(multi_timestep)}  {sorted(multi_timestep)}")
    print(f"  pinned for exactly one timestep:        {len(single_timestep)}  {sorted(single_timestep)}")
    print(f"  not pinned (oversized, size > budget):  {oversized_count}")
    print(f"  not pinned (lost FoM competition):      {len(tensors) - len(pinned_ids) - oversized_count}")


def validate_mobilenet_regression(result: dict) -> None:
    """
    MobileNet@2MB-specific regression checks against known, hand-derived
    facts about that one fixture (see module docstring) -- NOT general
    algorithm invariants, so only meaningful for that exact model/budget
    combination. Callers must gate this to that combination themselves.
    """
    tensors = result['tensors']
    assert len(tensors) == 30, f"expected 30 COSMA-tracked tensors, got {len(tensors)}"
    schedule_order = result['schedule_order']
    assert schedule_order == sorted(schedule_order), (
        f"expected identity schedule order on MobileNet's linear chain, got {schedule_order}")
    print("  [OK] MobileNet@2MB regression: 30 tensors tracked, identity schedule order")


def _model_name(model_json_path: str) -> str:
    """
    Short name a model's outputs are filed under -- its own directory
    name (e.g. '_exported/densenet/model.json' -> 'densenet'), or the
    bare filename stem if it has none / lives directly under onsram/ or
    cosma/. Same logic as cosma/run_experiments.py's own _model_name(),
    duplicated rather than imported -- that's a private helper of a
    top-level script, not part of any helpers/ module's stable contract.
    """
    parent = os.path.basename(os.path.dirname(os.path.abspath(model_json_path)))
    stem = os.path.splitext(os.path.basename(model_json_path))[0]
    return parent if parent and parent not in ('onsram', 'cosma', '_exported') else stem


def _log_file_name(model_json_path: str, spm_mb: float) -> str:
    """<model>_<budget>MB.log -- no timestamp: re-running the same
    (model, budget) overwrites its previous log rather than accumulating
    one file per run, same convention as
    cosma/run_experiments.py's own _log_file_name()."""
    budget_str = f"{spm_mb:g}".replace('.', 'p')
    return f"{_model_name(model_json_path)}_{budget_str}MB.log"


def _run_and_log(logs_dir, log_name: str, model_json_path: str, config_path: str,
                  memory_budget_bytes: int, run_mobilenet_regression: bool,
                  layer_stats: dict = None, bandwidth_bytes_per_cycle: float = None,
                  run_scale_sim: bool = True) -> Tuple[dict, dict]:
    """
    Runs one (model, budget) combination with its full verbose report
    (run_onsram()'s own pin-decision line, check_physical_validity()'s
    [OK] lines, validate_mobilenet_regression()'s line when applicable,
    print_detailed_report()'s schedule/FoM/pin-duration sections, and --
    when run_scale_sim -- print_dram_savings()'s real, engine-measured
    numbers) redirected straight into logs_dir/log_name instead of the
    terminal -- same discipline as cosma/run_experiments.py's own
    _run_and_log(). logs_dir=None sends the same report to the terminal
    instead (used by --no-logs). Re-raises on failure after appending the
    traceback to the same log file, so a failing combination still
    leaves something to look at.

    layer_stats: scale_sim_runner.run_baseline()'s output for this model,
        computed once by the caller and reused across every budget for the
        same model (see run_onsram_scale_sim()'s own docstring for why).
        Required whenever run_scale_sim is True.

    Returns (result, stats) -- stats merges check_physical_validity()'s
    own return value with run_onsram_scale_sim()'s (when run_scale_sim),
    reused by the caller to build a CSV summary row without recomputing.
    """
    header = (f"model: {model_json_path}\n"
              f"spm_budget_mb: {memory_budget_bytes / 1024 / 1024:.4f}\n"
              f"{'=' * 60}\n")
    log_path = os.path.join(logs_dir, log_name) if logs_dir else None
    log_file = None
    try:
        if log_path:
            os.makedirs(logs_dir, exist_ok=True)
            log_file = open(log_path, 'w', buffering=1)  # line-buffered, so `tail -f` works live
            log_file.write(header)
        cm = contextlib.redirect_stdout(log_file) if log_file is not None else contextlib.nullcontext()
        with cm:
            result = run_onsram(model_json_path, memory_budget_bytes, verbose=True)
            stats = check_physical_validity(result, memory_budget_bytes, verbose=True)
            if run_mobilenet_regression:
                validate_mobilenet_regression(result)
            print_detailed_report(result, memory_budget_bytes)
            if run_scale_sim:
                # verbose=False: SCALE-Sim's own per-layer tqdm progress bar
                # writes to stderr by default, bypassing this function's own
                # stdout redirect entirely -- so verbose=True here doesn't
                # even end up in the log file, it just floods the terminal
                # (and costs real wall-clock time writing it) for no benefit;
                # print_dram_savings() below already reports what matters.
                scale_sim_result = run_onsram_scale_sim(
                    result, model_json_path, config_path, memory_budget_bytes,
                    layer_stats, bandwidth_bytes_per_cycle, verbose=True)
                print_dram_savings(scale_sim_result)
                stats = {**stats, **scale_sim_result}
        return result, stats
    except Exception:
        if log_file is not None:
            log_file.write(f"\n{traceback.format_exc()}\n")
            log_file.flush()
        raise
    finally:
        if log_file is not None:
            log_file.close()


def _summary_row(model_arg: str, spm_mb: float, stats: dict, log_path: str) -> dict:
    row = {
        'model': model_arg, 'spm_mb': spm_mb, 'status': 'OK',
        'pinned': stats['pinned_count'], 'total_tensors': stats['total_tensors'],
        'peak_mb': round(stats['peak_bytes'] / 1024 / 1024, 4),
        'oversized': stats['oversized_count'],
        'dram_reduction_pct': '', 'speedup': '',
        'log': log_path or '', 'error': '',
    }
    if 'dram_traffic_reduction_pct' in stats:
        row['dram_reduction_pct'] = round(stats['dram_traffic_reduction_pct'], 2)
        row['speedup'] = round(stats['speedup'], 4)
    return row


def _error_row(model_arg: str, spm_mb: float, error: Exception, log_path: str) -> dict:
    return {
        'model': model_arg, 'spm_mb': spm_mb, 'status': 'ERROR',
        'pinned': '', 'total_tensors': '', 'peak_mb': '', 'oversized': '',
        'dram_reduction_pct': '', 'speedup': '',
        'log': log_path or '', 'error': f"{type(error).__name__}: {error}",
    }


def print_table(rows: list) -> None:
    cols = ['model', 'spm_mb', 'status', 'pinned', 'total_tensors', 'peak_mb',
            'oversized', 'dram_reduction_pct', 'speedup', 'error']
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print('  '.join(c.ljust(widths[c]) for c in cols))
    print('  '.join('-' * widths[c] for c in cols))
    for r in rows:
        print('  '.join(str(r[c]).ljust(widths[c]) for c in cols))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="OnSRAM-Static: FoM scoring + greedy whole-interval pinning "
                    "+ BFS-DFS scheduling on a real model.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='append', dest='models', default=None,
                        help="Model to run: a model.json path, a .tflite path, or a bare "
                             "name under cosma/_exported/<name>/model.json. Repeatable to "
                             "sweep multiple models. Defaults to MobileNet.")
    parser.add_argument('--spm-mb', type=float, nargs='+', default=[DEFAULT_SPM_MB],
                        help=f"SPM budget(s) in MB to sweep (default: {DEFAULT_SPM_MB}).")
    parser.add_argument('--plot', action='store_true',
                        help="Save an SPM-occupancy-over-time PNG for each (model, budget) "
                             f"run under {DEFAULT_PLOT_DIR}/.")
    parser.add_argument('--plot-dir', default=DEFAULT_PLOT_DIR,
                        help=f"Directory to save plots into when --plot is set (default: {DEFAULT_PLOT_DIR}).")
    parser.add_argument('--no-logs', action='store_true',
                        help="Print each combination's full verbose report to the terminal "
                             "instead of saving it to a per-combination log file.")
    parser.add_argument('--logs-dir', default=DEFAULT_LOGS_DIR,
                        help=f"Directory for per-combination log files (default: {DEFAULT_LOGS_DIR}).")
    parser.add_argument('--out-csv', default=None,
                        help="Also write a cross-combination summary CSV to this path.")
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        help=f"SCALE-Sim hardware config file (default: {DEFAULT_CONFIG}).")
    parser.add_argument('--no-scale-sim', action='store_true',
                        help="Skip Phase D's real SCALE-Sim pass (scale_sim_runner.run_onsram_aware()) "
                             "and only run the fast pinning decision -- no DRAM-savings numbers.")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    model_args = args.models or [DEFAULT_MODEL]
    logs_dir = None if args.no_logs else args.logs_dir
    run_scale_sim = not args.no_scale_sim

    if run_scale_sim and not os.path.isfile(args.config):
        # configparser.read() silently ignores a missing file rather than
        # raising -- without this check, a bad --config path (e.g. a
        # relative path resolved against the wrong cwd) surfaces many
        # steps later as a confusing "NoSectionError: No section: 'general'"
        # instead of a clear, immediate, actionable message.
        sys.exit(f"error: --config file not found: {args.config}\n"
                 f"(if this is a relative path, note it resolves against your current "
                 f"directory, not the repo root or onsram/ -- the default "
                 f"({DEFAULT_CONFIG}) is always an absolute path and doesn't have this issue)")

    if args.plot:
        os.makedirs(args.plot_dir, exist_ok=True)

    rows = []
    for model_arg in model_args:
        try:
            model_json_path = resolve_model_arg(model_arg)
        except Exception as e:  # noqa: BLE001 -- one bad model shouldn't abort the whole sweep
            print(f"  FAILED to resolve model={model_arg}: {type(e).__name__}: {e}", flush=True)
            rows.extend(_error_row(model_arg, spm_mb, e, log_path=None) for spm_mb in args.spm_mb)
            continue
        model_label = os.path.basename(model_arg.rstrip('/'))

        layer_stats = None
        bandwidth_bytes_per_cycle = None
        if run_scale_sim:
            # Schedule- and budget-independent (see run_onsram_scale_sim()'s
            # docstring) -- computed once per model here, reused for every
            # spm_mb below, instead of re-running SCALE-Sim's plain baseline
            # pass once per budget.
            print(f"Running SCALE-Sim baseline for {model_arg} ...", file=sys.stderr, flush=True)
            heartbeat_stop = _start_heartbeat(f"SCALE-Sim baseline for {model_arg}")
            try:
                layer_stats = scale_sim_runner.run_baseline(model_json_path, args.config, verbose=True)
                bandwidth_bytes_per_cycle = _default_bandwidth_words_per_cycle(args.config)
            except Exception as e:  # noqa: BLE001 -- one bad model shouldn't abort the whole sweep
                print(f"  FAILED SCALE-Sim baseline for {model_arg}: {type(e).__name__}: {e}", flush=True)
                rows.extend(_error_row(model_arg, spm_mb, e, log_path=None) for spm_mb in args.spm_mb)
                continue
            finally:
                heartbeat_stop.set()

        for spm_mb in args.spm_mb:
            memory_budget_bytes = int(spm_mb * 1024 * 1024)
            log_name = _log_file_name(model_json_path, spm_mb)
            log_path = os.path.join(logs_dir, log_name) if logs_dir else None
            is_mobilenet_regression = (model_arg == DEFAULT_MODEL and spm_mb == DEFAULT_SPM_MB)

            print(f"Running: model={model_arg} spm={spm_mb:g}MB ...", file=sys.stderr, flush=True)
            try:
                result, stats = _run_and_log(
                    logs_dir, log_name, model_json_path, args.config, memory_budget_bytes,
                    run_mobilenet_regression=is_mobilenet_regression,
                    layer_stats=layer_stats, bandwidth_bytes_per_cycle=bandwidth_bytes_per_cycle,
                    run_scale_sim=run_scale_sim)
                rows.append(_summary_row(model_arg, spm_mb, stats, log_path))
                dram_note = (f", DRAM -{stats['dram_traffic_reduction_pct']:.1f}%, "
                             f"speedup {stats['speedup']:.3f}x"
                             if 'dram_traffic_reduction_pct' in stats else "")
                print(f"  {stats['pinned_count']}/{stats['total_tensors']} pinned, "
                      f"peak {stats['peak_bytes'] / 1024 / 1024:.4f} MB / {spm_mb:g} MB"
                      f"{dram_note}"
                      + (f"  (log: {log_path})" if log_path else ""), flush=True)
            except Exception as e:  # noqa: BLE001 -- one bad combination shouldn't abort the sweep
                rows.append(_error_row(model_arg, spm_mb, e, log_path))
                print(f"  FAILED: {type(e).__name__}: {e}"
                      + (f"  (log: {log_path})" if log_path else ""), flush=True)
                continue

            if args.plot:
                out_path = os.path.join(args.plot_dir, f"{model_label}_{spm_mb:g}MB.png")
                visualize.plot_spm_usage(
                    result['tensors'], result['resident_action'], result['spm_plan'],
                    memory_budget_bytes, schedule_len=len(result['schedule']),
                    out_path=out_path,
                    title=f"OnSRAM-Static SPM occupancy: {model_label} @ {spm_mb:g} MB")
                print(f"  [plot] saved {out_path}")

    print()
    print_table(rows)

    if args.out_csv:
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out_csv}")
