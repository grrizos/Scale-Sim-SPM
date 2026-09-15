# cosma/run_paper_baselines.py
"""
Implements the COSMA paper's own comparison baselines (§V-A2): TensorFlow-
Lite's linear allocator combined with {default schedule, MPMF schedule} x
{Belady's algorithm, ILP-based greedy replacement} -- 4 combinations, each
run through the exact same real SCALE-Sim engine and accounting COSMA's
own numbers use, so the resulting non-compulsory-byte numbers are directly
comparable to COSMA's own (not a different measurement) -- see
docs/baseline_construction.md.

Mirrors run_cosma.py's orchestration pattern deliberately (same
resolve -> load_graph -> ILP/policy -> baseline sim -> COSMA-aware sim ->
combine shape), but is a fully separate, additive file. In particular, the
idealized-vs-real accounting block below (_account()) is a deliberate,
disclosed DUPLICATE of run_cosma.py's own (not an import of a shared
helper) -- extracting a shared helper would require editing run_cosma.py,
which this effort must not touch while it may be in parallel use
elsewhere. See docs/baseline_construction.md's "Design" section for the
full isolation rationale.

New building blocks used (all new, additive files from this same effort):
  helpers/schedule_variants.py      -- default & MPMF schedules
  helpers/replacement_engine.py     -- shared WHAT/WHEN simulation loop
  helpers/belady_policy.py          -- furthest-future eviction
  helpers/ilp_greedy_policy.py      -- local per-decision ILP eviction
  helpers/tflite_arena_allocator.py -- WHERE (real TFLite placement algorithm)

Existing, unmodified building blocks reused:
  spm_common/model_resolver.resolve_model_json()
  spm_common/graph_builder.load_graph()
  helpers/baseline.run_baseline() / run_cosma_aware()
  helpers/cosma_Ilp.assert_tensors_fit_budget() / compute_true_mpmf_bytes()
  run_cosma.run_cosma()  -- optional 5th "cosma_native" comparison row
"""
import argparse
import contextlib
import csv
import functools
import os
import sys
import threading
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from spm_common import graph_builder
from helpers import baseline
from helpers import cosma_Ilp
from spm_common import model_resolver
from helpers import schedule_variants
from helpers import replacement_engine
from helpers import belady_policy
from helpers import ilp_greedy_policy
from helpers import tflite_arena_allocator
import run_cosma
import visualize_spm

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), 'configs', 'scale.cfg')
DEFAULT_RESULTS_DIR = os.path.join(HERE, 'results')
DEFAULT_LOGS_DIR = os.path.join(HERE, 'logs')

CONV_LIKE_OPS = ('CONV2D', 'DEPTHWISE_CONV2D')

POLICY_FNS = {
    'belady': belady_policy.choose_victims,
    'ilp_greedy': ilp_greedy_policy.choose_victims,
}

_HEARTBEAT_INTERVAL_SEC = 20


def _start_heartbeat(label: str) -> threading.Event:
    """
    Prints a short "still running" line to stderr every
    _HEARTBEAT_INTERVAL_SEC seconds for as long as the returned Event
    stays unset -- same liveness-signal pattern as
    run_experiments.py's own _start_heartbeat(), duplicated (not
    imported) rather than shared, matching this file's own "fully
    separate, additive file" isolation rationale (see module docstring).
    Caller must call .set() on the returned Event once the covered step
    finishes.
    """
    stop = threading.Event()
    start = time.monotonic()

    def _beat():
        while not stop.wait(_HEARTBEAT_INTERVAL_SEC):
            elapsed = time.monotonic() - start
            print(f"[heartbeat] still running: {label} elapsed={elapsed:.0f}s",
                  file=sys.stderr)

    threading.Thread(target=_beat, daemon=True).start()
    return stop


def _model_name(model_json_path: str) -> str:
    """Same short-name convention as run_experiments.py's own
    _model_name() / visualize_spm.default_out_path()'s inline logic --
    duplicated locally (see module docstring's isolation rationale), used
    to build default log/CSV filenames that agree with the plot filenames
    visualize_spm.default_out_path() already produces for this model."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(model_json_path)))
    stem = os.path.splitext(os.path.basename(model_json_path))[0]
    return parent if parent and parent != 'cosma' else stem


def _array_dims_tag(config_path: str) -> str:
    """'<ArrayHeight>x<ArrayWidth>' from a scale.cfg -- duplicate of
    run_cosma.py's own tag (see module docstring's isolation rationale),
    used to keep default log/CSV/plot filenames from silently overwriting
    each other across different array configs."""
    from scalesim.scale_config import scale_config
    config = scale_config()
    config.read_conf_file(config_path)
    arr_row, arr_col = config.get_array_dims()
    return f"{arr_row}x{arr_col}"


class _Tee:
    """
    Writes to both a live stream (the real terminal) and a log file,
    flushing both after every write -- unlike run_experiments.py's own
    _run_and_log() (which redirects to file only, to avoid flooding a
    multi-combination terminal sweep), this file's own output is wanted
    live AND saved every time, so this duplicates rather than redirects.
    """
    def __init__(self, live_stream, log_file):
        self.live_stream = live_stream
        self.log_file = log_file

    def write(self, s):
        self.live_stream.write(s)
        self.live_stream.flush()
        self.log_file.write(s)
        self.log_file.flush()

    def flush(self):
        self.live_stream.flush()
        self.log_file.flush()


def _default_bandwidth_words_per_cycle(config_path: str) -> float:
    """Same estimate run_cosma.py's own _default_bandwidth_words_per_cycle()
    uses -- re-derived locally (not imported) to avoid any coupling to
    run_cosma.py beyond its public run_cosma() function."""
    from scalesim.scale_config import scale_config
    config = scale_config()
    config.read_conf_file(config_path)
    if config.use_user_dram_bandwidth():
        return float(config.get_bandwidths_as_list()[0])
    _, arr_col = config.get_array_dims()
    return float(arr_col)


def _account(nodes, tensors, resident_action, schedule, layer_stats, plan_stats,
             bandwidth_bytes_per_cycle) -> dict:
    """
    Deliberate duplicate of run_cosma.py's own idealized-vs-real
    accounting block (see this module's docstring for why) -- same logic,
    applied to a TFLite-linear-allocator + replacement-policy plan instead
    of COSMA's own ILP plan. resident_action here can legitimately spill
    then retrieve the SAME tensor more than once (unlike COSMA's own
    Eq.8-constrained "at most one spill per tensor" output) -- this block
    handles that identically, since it only ever looks at one (a, t) pair
    at a time and never assumes at most one S per tensor overall.
    """
    baseline_total_cycles = 0
    cosma_total_cycles = 0
    baseline_dram_bytes = 0
    cosma_dram_bytes = 0
    total_idealized_spill_bytes = 0
    total_idealized_retrieve_bytes = 0
    total_real_retrieve_bytes = 0
    total_ifmap_residency_credit_bytes = 0
    total_ofmap_residency_credit_bytes = 0
    layer_bound_breakdown = []

    A = sorted(tensors.keys())
    for t, lid in schedule:
        base_s = layer_stats[lid]
        plan_s = plan_stats[lid]
        node = nodes[lid]

        is_conv_like = node.op in CONV_LIKE_OPS
        real_retrieve_tensor = (
            node.activation_inputs[0]
            if is_conv_like and node.activation_inputs
               and resident_action.get((node.activation_inputs[0], t)) == 'R'
            else None
        )

        idealized_bytes = sum(
            tensors[a].size_bytes for a in A
            if resident_action.get((a, t)) in ('S', 'R') and a != real_retrieve_tensor
        )
        idealized_spill_bytes = sum(
            tensors[a].size_bytes for a in A if resident_action.get((a, t)) == 'S')
        total_idealized_spill_bytes += idealized_spill_bytes
        total_idealized_retrieve_bytes += idealized_bytes - idealized_spill_bytes
        if real_retrieve_tensor is not None:
            total_real_retrieve_bytes += plan_s['ifmap_dram_bytes']

        total_ifmap_residency_credit_bytes += (
            base_s['ifmap_dram_bytes'] - plan_s['ifmap_dram_bytes'])
        total_ofmap_residency_credit_bytes += (
            base_s['ofmap_dram_bytes'] - plan_s['ofmap_dram_bytes'])

        baseline_compulsory = (base_s['ifmap_dram_bytes'] + base_s['ofmap_dram_bytes']
                                + base_s['filter_dram_bytes'])
        plan_compulsory = (plan_s['ifmap_dram_bytes'] + plan_s['ofmap_dram_bytes']
                            + plan_s['filter_dram_bytes'] + idealized_bytes)

        baseline_mem_cycles = baseline_compulsory / bandwidth_bytes_per_cycle
        plan_mem_cycles = plan_compulsory / bandwidth_bytes_per_cycle
        layer_bound_breakdown.append({
            't': t, 'layer_id': lid, 'op': node.op,
            'baseline_bound': 'memory' if baseline_mem_cycles > base_s['compute_cycles'] else 'compute',
            'plan_bound': 'memory' if plan_mem_cycles > plan_s['compute_cycles'] else 'compute',
        })

        baseline_total_cycles += max(base_s['compute_cycles'], baseline_mem_cycles)
        cosma_total_cycles += max(plan_s['compute_cycles'], plan_mem_cycles)
        baseline_dram_bytes += baseline_compulsory
        cosma_dram_bytes += plan_compulsory

    total_non_compulsory_access_bytes = (
        total_idealized_spill_bytes + total_idealized_retrieve_bytes + total_real_retrieve_bytes)

    dram_traffic_reduction_pct = (
        100.0 * (baseline_dram_bytes - cosma_dram_bytes) / baseline_dram_bytes
        if baseline_dram_bytes > 0 else 0.0)
    speedup = (baseline_total_cycles / cosma_total_cycles
               if cosma_total_cycles > 0 else float('inf'))

    return {
        'baseline_total_cycles': baseline_total_cycles,
        'cosma_total_cycles': cosma_total_cycles,
        'baseline_dram_bytes': baseline_dram_bytes,
        'cosma_dram_bytes': cosma_dram_bytes,
        'total_idealized_spill_bytes': total_idealized_spill_bytes,
        'total_idealized_retrieve_bytes': total_idealized_retrieve_bytes,
        'total_real_retrieve_bytes': total_real_retrieve_bytes,
        'total_non_compulsory_access_bytes': total_non_compulsory_access_bytes,
        'total_ifmap_residency_credit_bytes': total_ifmap_residency_credit_bytes,
        'total_ofmap_residency_credit_bytes': total_ofmap_residency_credit_bytes,
        'dram_traffic_reduction_pct': dram_traffic_reduction_pct,
        'speedup': speedup,
        'layer_bound_breakdown': layer_bound_breakdown,
    }


def run_paper_baseline(nodes, tensors, model_json_path: str, config_path: str,
                        schedule, memory_budget_bytes: int, choose_victims_fn,
                        layer_stats: dict, bandwidth_bytes_per_cycle: float,
                        ilp_greedy_time_limit_sec: float = 10,
                        ilp_greedy_solver: str = 'cbc',
                        verbose: bool = True,
                        combo: str = None,
                        baseline_action: dict = None,
                        array_tag: str = None,
                        save_plot: bool = True) -> dict:
    """
    Runs ONE (schedule, policy) combination end-to-end:
    replacement_engine.simulate_replacement() ->
    tflite_arena_allocator.place_tensors_linear() ->
    helpers.baseline.run_cosma_aware() (unmodified, real SCALE-Sim) ->
    _account() (see module docstring). Returns a summary dict shaped like
    run_cosma.run_cosma()'s own (same key names for the metrics the two
    share), for direct side-by-side comparability.

    May raise tflite_arena_allocator.TfliteArenaAllocationError (placement
    couldn't fit -- a real, reportable finding about TFLite's allocator,
    not necessarily a bug) or replacement_engine.ReplacementInfeasible
    (budget below what this schedule/policy can achieve) -- callers
    sweeping many combinations should catch both, see
    run_all_paper_baselines().

    save_plot: if True (default) and combo/baseline_action are given,
    saves a baseline-vs-plan occupancy PNG via visualize_spm's own
    generic render_comparison() -- same plotting code run_cosma.py's own
    cosma_native row uses, reused here since this combo's spm_plan/
    resident_action are in the identical generic shape. Filed under
    cosma/spm_plots/, tagged with `combo` (e.g. 'default+belady') instead
    of run_cosma.py's 'S'/'D' schedule tag, so all 5 rows' plots for the
    same (model, budget) sit side by side without overwriting each other.
    """
    victims_fn = choose_victims_fn
    if choose_victims_fn is ilp_greedy_policy.choose_victims:
        victims_fn = functools.partial(
            ilp_greedy_policy.choose_victims,
            time_limit_sec=ilp_greedy_time_limit_sec, solver=ilp_greedy_solver)

    resident_action = replacement_engine.simulate_replacement(
        nodes, tensors, schedule, memory_budget_bytes, victims_fn)
    spm_plan = tflite_arena_allocator.place_tensors_linear(
        tensors, resident_action, memory_budget_bytes)
    plan_stats = baseline.run_cosma_aware(
        model_json_path, config_path, resident_action, spm_plan=spm_plan,
        tensors=tensors, memory_budget_bytes=memory_budget_bytes,
        schedule=schedule, verbose=verbose)

    accounting = _account(nodes, tensors, resident_action, schedule,
                           layer_stats, plan_stats, bandwidth_bytes_per_cycle)
    summary = {'status': 'Optimal', 'memory_budget_bytes': memory_budget_bytes,
               'spm_plan': spm_plan, 'resident_action': resident_action,
               # dict(schedule), not the {t: t} identity render_comparison()
               # falls back to -- under the 'mpmf' schedule t != layer id,
               # so the fallback would mislabel the plot's x-axis.
               'schedule_layer_at_t': dict(schedule)}
    summary.update(accounting)
    if verbose:
        print(f"  non-compulsory bytes: {summary['total_non_compulsory_access_bytes']}, "
              f"DRAM reduction: {summary['dram_traffic_reduction_pct']:.1f}%, "
              f"speedup: {summary['speedup']:.4f}x")

    if save_plot and combo is not None and baseline_action is not None:
        out_path = visualize_spm.default_out_path(
            model_json_path, memory_budget_bytes, array_tag=array_tag, schedule_tag=combo)
        visualize_spm.render_comparison(
            nodes, tensors, baseline_action, summary, memory_budget_bytes, out_path,
            title=f"{os.path.basename(model_json_path)} -- {combo} @ "
                  f"{memory_budget_bytes / 1024:.2f} KB")
        if verbose:
            print(f"  saved plot to {out_path}")

    return summary


def run_all_paper_baselines(model_json_path: str, config_path: str = DEFAULT_CONFIG,
                             memory_budget_bytes: int = 128 * 1024,
                             include_cosma_native: bool = True,
                             ilp_time_limit_sec: float = None,
                             ilp_greedy_time_limit_sec: float = 10,
                             solver: str = 'cbc',
                             exporter: str = model_resolver.DEFAULT_EXPORTER,
                             export_dir: str = model_resolver.DEFAULT_EXPORT_DIR,
                             force_export: bool = False,
                             verbose: bool = True,
                             save_plots: bool = True) -> dict:
    """
    Loads the graph once, computes layer_stats via baseline.run_baseline()
    once (schedule-independent -- reused across all combinations below,
    same optimization run_experiments.py already exploits across a BUDGET
    sweep, extended here across the SCHEDULE axis too, and passed through
    to run_cosma.run_cosma() for the cosma_native row too), computes both
    schedule variants once, then runs all 4 (schedule, policy)
    combinations plus (if include_cosma_native) COSMA's own real pipeline
    via run_cosma.run_cosma() for a direct, same-budget comparison row.
    Net cost: 6 real SCALE-Sim passes per (model, budget) -- 1 shared
    baseline + 4 new-baseline run_cosma_aware() calls + 1 cosma_native
    run_cosma_aware() call (inside run_cosma.run_cosma()) -- not 7+.

    A combination that fails (TfliteArenaAllocationError -- placement
    fragmentation, or ReplacementInfeasible -- budget below what that
    policy/schedule can achieve) is recorded as an explicit error entry,
    not silently dropped or allowed to abort the other combinations --
    same "one bad combination shouldn't abort the batch" spirit as
    run_experiments.py's own _error_row() pattern.

    ilp_time_limit_sec: applied to BOTH real ILP solves in this function
    -- the MPMF schedule solve (cosma_Ilp.compute_true_mpmf_bytes()) and,
    if include_cosma_native, COSMA's own full joint ILP solve (inside
    run_cosma.run_cosma()). Previously only bounded the former, leaving
    the cosma_native row's solve silently unbounded regardless of this
    argument -- confirmed directly: a real run left it running 500+s with
    no limit applied while --time-limit was set to 1500s. Does NOT apply
    to ilp_greedy_time_limit_sec's many small per-decision solves, which
    intentionally stay on their own separate, much shorter default (see
    that argument's own docstring).

    Returns {'default+belady', 'default+ilp_greedy', 'mpmf+belady',
    'mpmf+ilp_greedy'[, 'cosma_native']: summary dict (or
    {'status': 'ERROR', 'error': str} on failure)}.
    """
    model_json_path = model_resolver.resolve_model_json(
        model_json_path, exporter, export_dir, force_export)
    nodes, tensors = graph_builder.load_graph(model_json_path)
    cosma_Ilp.assert_tensors_fit_budget(tensors, memory_budget_bytes)

    bandwidth_bytes_per_cycle = _default_bandwidth_words_per_cycle(config_path)
    # Structural, schedule-independent -- same one baseline_action used for
    # every combo's plot below (mirrors layer_stats's own "compute once,
    # reuse across combos" pattern), and for the cosma_native plot's own
    # baseline panel (run_cosma.py computes an identical one internally).
    baseline_action = visualize_spm.compute_baseline_resident_action(nodes, tensors)
    array_tag = _array_dims_tag(config_path)

    if verbose:
        print(f"Running SCALE-Sim baseline for {model_json_path} ...")
    # verbose=verbose enables SCALE-Sim's own per-layer tqdm progress (see
    # run_cosma.py's identical wiring) -- otherwise a large model's
    # baseline pass produces zero output until it finishes. The heartbeat
    # underneath is still needed: this is a single call for the whole
    # model (not per-combination), so it isn't covered by any
    # per-combination signal below.
    heartbeat_stop = _start_heartbeat(f"SCALE-Sim baseline for {model_json_path}")
    try:
        layer_stats = baseline.run_baseline(model_json_path, config_path, verbose=verbose)
    finally:
        heartbeat_stop.set()

    default_sched = schedule_variants.default_operator_schedule(nodes)
    if verbose:
        print("Solving MPMF schedule ILP (cosma_Ilp.compute_true_mpmf_bytes) ...")
    # msg=verbose surfaces Gurobi/CBC's own native solve log -- this is a
    # real, potentially slow ILP solve (see compute_true_mpmf_bytes()'s
    # own docstring), untested at DenseNet-121/ImageNet scale through
    # this file before, so a heartbeat covers it too in case the solver
    # itself goes quiet for a while (e.g. during presolve).
    heartbeat_stop = _start_heartbeat("MPMF schedule ILP solve")
    try:
        _, mpmf_tensor_sched = cosma_Ilp.compute_true_mpmf_bytes(
            nodes, tensors, time_limit_sec=ilp_time_limit_sec, solver=solver, msg=verbose)
    finally:
        heartbeat_stop.set()
    mpmf_sched = schedule_variants.mpmf_operator_schedule(nodes, tensors, mpmf_tensor_sched)

    schedules = {'default': default_sched, 'mpmf': mpmf_sched}
    results = {}
    for sched_name, sched in schedules.items():
        for policy_name, policy_fn in POLICY_FNS.items():
            combo = f"{sched_name}+{policy_name}"
            if verbose:
                print(f"\n=== {combo} ===")
            # Covers both the per-decision ILP-greedy replacement loop
            # (many small solves, each up to ilp_greedy_time_limit_sec)
            # and the real SCALE-Sim run_cosma_aware() pass below it --
            # either can run long with nothing else printed in between.
            heartbeat_stop = _start_heartbeat(f"{combo} @ {memory_budget_bytes} bytes")
            try:
                results[combo] = run_paper_baseline(
                    nodes, tensors, model_json_path, config_path, sched,
                    memory_budget_bytes, policy_fn, layer_stats,
                    bandwidth_bytes_per_cycle,
                    ilp_greedy_time_limit_sec=ilp_greedy_time_limit_sec,
                    ilp_greedy_solver=solver, verbose=verbose,
                    combo=combo, baseline_action=baseline_action,
                    array_tag=array_tag, save_plot=save_plots)
            except (tflite_arena_allocator.TfliteArenaAllocationError,
                    replacement_engine.ReplacementInfeasible) as e:
                results[combo] = {'status': 'ERROR', 'error': f"{type(e).__name__}: {e}"}
                if verbose:
                    print(f"  FAILED: {type(e).__name__}: {e}")
            finally:
                heartbeat_stop.set()

    if include_cosma_native:
        if verbose:
            print("\n=== cosma_native ===")
        heartbeat_stop = _start_heartbeat(f"cosma_native @ {memory_budget_bytes} bytes")
        try:
            # plot_out_path override so this plot is tagged 'cosma_native'
            # like its 4 sibling combos' plots, instead of run_cosma.py's
            # own generic 'S'/'D' schedule tag (this file never exposes
            # free_schedule, so it'd always be 'S' -- less identifiable
            # sitting next to 'default+belady' etc. in the same directory).
            cosma_native_plot_path = (
                visualize_spm.default_out_path(
                    model_json_path, memory_budget_bytes,
                    array_tag=array_tag, schedule_tag='cosma_native')
                if save_plots else None)
            results['cosma_native'] = run_cosma.run_cosma(
                model_json_path=model_json_path, config_path=config_path,
                memory_budget_bytes=memory_budget_bytes, layer_stats=layer_stats,
                bandwidth_bytes_per_cycle=bandwidth_bytes_per_cycle,
                ilp_time_limit_sec=ilp_time_limit_sec,
                verbose=verbose, save_plot=save_plots,
                plot_out_path=cosma_native_plot_path, solver=solver)
            if verbose:
                r = results['cosma_native']
                print(f"  non-compulsory bytes: {r['total_non_compulsory_access_bytes']}, "
                      f"DRAM reduction: {r['dram_traffic_reduction_pct']:.1f}%, "
                      f"speedup: {r['speedup']:.4f}x")
        except Exception as e:  # noqa: BLE001 -- one failed combination shouldn't abort the rest
            results['cosma_native'] = {'status': 'ERROR', 'error': f"{type(e).__name__}: {e}"}
            if verbose:
                print(f"  FAILED: {type(e).__name__}: {e}")
        finally:
            heartbeat_stop.set()

    return results


def print_comparison_table(results: dict, budget_kb: float) -> None:
    cols = ['combo', 'status', 'total_non_compulsory_access_bytes',
            'dram_traffic_reduction_pct', 'speedup', 'error']
    rows = []
    for combo, r in results.items():
        rows.append({
            'combo': combo,
            'status': r.get('status', 'Optimal'),
            'total_non_compulsory_access_bytes': r.get('total_non_compulsory_access_bytes', ''),
            'dram_traffic_reduction_pct':
                round(r['dram_traffic_reduction_pct'], 2) if 'dram_traffic_reduction_pct' in r else '',
            'speedup': round(r['speedup'], 4) if 'speedup' in r else '',
            'error': r.get('error', ''),
        })
    widths = {c: max(len(c), *(len(str(row[c])) for row in rows)) for c in cols}
    print(f"\n--- Budget: {budget_kb} KB ---")
    print('  '.join(c.ljust(widths[c]) for c in cols))
    print('  '.join('-' * widths[c] for c in cols))
    for row in rows:
        print('  '.join(str(row[c]).ljust(widths[c]) for c in cols))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-json', required=True,
                         help='Path to model.json, or a raw .tflite file.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--budgets-kb', nargs='+', type=float, required=True,
                         help='One or more SPM budgets in KB.')
    parser.add_argument('--exporter', default=model_resolver.DEFAULT_EXPORTER)
    parser.add_argument('--export-dir', default=model_resolver.DEFAULT_EXPORT_DIR)
    parser.add_argument('--force-export', action='store_true')
    parser.add_argument('--time-limit', type=float, default=1500,
                         help='Time limit (seconds) applied to BOTH real ILP solves: the '
                              'MPMF schedule solve and, unless --no-cosma-native, the '
                              'cosma_native row\'s own full joint COSMA ILP solve. Default: '
                              '1500s -- with solver=gurobi (the default), a solve that hits '
                              'this limit still returns its best feasible incumbent instead '
                              'of failing (see cosma_Ilp.solve()\'s has_feasible_incumbent '
                              'docstring). Does NOT apply to --ilp-greedy-time-limit\'s many '
                              'small per-decision solves. Pass a larger value, or edit this '
                              'default back to None, for unbounded.')
    parser.add_argument('--ilp-greedy-time-limit', type=float, default=10,
                         help='Per-decision-point time limit (seconds) for the local '
                              'ILP-greedy replacement policy solve. Default: 10.')
    parser.add_argument('--solver', choices=['cbc', 'gurobi'], default='gurobi',
                         help='ILP solver for the MPMF schedule and ILP-greedy replacement '
                              '(default: cbc, no license needed). See cosma_Ilp.solve().')
    parser.add_argument('--no-cosma-native', action='store_true',
                         help="Skip the COSMA-native comparison row (run_cosma.run_cosma()).")
    parser.add_argument('--no-plots', action='store_true',
                         help="Skip saving each combination's baseline-vs-plan occupancy PNG "
                              "(saved to cosma/spm_plots/ by default, one per (budget, combo)).")
    parser.add_argument('--out-csv', default=None,
                         help='Path to write a combined results CSV across all budgets. '
                              'Defaults to cosma/results/<model>_<array>_paper_baselines.csv -- '
                              'always written, this only lets you pick where.')
    parser.add_argument('--logs-dir', default=None,
                         help='Directory to save the full run log (everything printed, '
                              'including the ILP solvers\' own native output) to. Defaults to '
                              'cosma/logs/ -- always written, this only lets you pick where.')
    parser.add_argument('--no-logs', action='store_true',
                         help='Skip saving the run log; print only.')
    args = parser.parse_args()

    array_tag = _array_dims_tag(args.config)
    name_tag = f"{_model_name(args.model_json)}_{array_tag}_paper_baselines"

    log_path = None
    if not args.no_logs:
        logs_dir = args.logs_dir or DEFAULT_LOGS_DIR
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"{name_tag}.log")

    log_file = open(log_path, 'w', buffering=1) if log_path else None
    try:
        stdout_target = _Tee(sys.stdout, log_file) if log_file is not None else sys.stdout
        with contextlib.redirect_stdout(stdout_target):
            all_rows = []
            for budget_kb in args.budgets_kb:
                results = run_all_paper_baselines(
                    model_json_path=args.model_json, config_path=args.config,
                    memory_budget_bytes=int(budget_kb * 1024),
                    include_cosma_native=not args.no_cosma_native,
                    ilp_time_limit_sec=args.time_limit,
                    ilp_greedy_time_limit_sec=args.ilp_greedy_time_limit,
                    solver=args.solver, exporter=args.exporter, export_dir=args.export_dir,
                    force_export=args.force_export, verbose=True,
                    save_plots=not args.no_plots)
                print_comparison_table(results, budget_kb)
                for combo, r in results.items():
                    all_rows.append({
                        'model': args.model_json, 'budget_kb': budget_kb, 'combo': combo,
                        'status': r.get('status', 'Optimal'),
                        'total_non_compulsory_access_bytes':
                            r.get('total_non_compulsory_access_bytes', ''),
                        'dram_traffic_reduction_pct': r.get('dram_traffic_reduction_pct', ''),
                        'speedup': r.get('speedup', ''),
                        'error': r.get('error', ''),
                    })

            out_csv = args.out_csv or os.path.join(DEFAULT_RESULTS_DIR, f"{name_tag}.csv")
            out_dir = os.path.dirname(out_csv)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'model', 'budget_kb', 'combo', 'status',
                    'total_non_compulsory_access_bytes', 'dram_traffic_reduction_pct',
                    'speedup', 'error'])
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"\nWrote {len(all_rows)} rows to {out_csv}")
            if log_path:
                print(f"Wrote full run log to {log_path}")
    finally:
        if log_file is not None:
            log_file.close()
