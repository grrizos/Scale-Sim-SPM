# cosma/visualize_spm.py
"""
Debug visualization + fast budget-range finder for COSMA's SPM plan.
Renders one PNG with two stacked panels, sharing a timestep x-axis:

  - top:    "Baseline (no cross-layer residency)" -- what SCALE-Sim's
            unmodified engine actually does with no COSMA involved.
  - bottom: "COSMA plan @ <budget>KB" -- the ILP's chosen SPM placement.

Only graph_builder.load_graph() + cosma_Ilp.{build_cosma_model, solve,
extract_results} are used -- no baseline.run_baseline(), no
run_cosma_aware(), no SCALE-Sim at all. Both panels are derivable from the
graph/ILP alone, so this stays fast (milliseconds to a few seconds) even
for ResNet-50/Inception-V3-sized graphs, unlike run_cosma.py/
run_experiments.py which drive real (slow) SCALE-Sim passes. Use --bounds-
only to skip the ILP solve entirely and just get the two numbers that
matter for picking a budget worth testing at all.

Why the baseline panel has no SPM address axis: SCALE-Sim's default
double_buffered_scratchpad uses three independent, fixed-size typed
buffers (ifmap/filter/ofmap SRAM, sized from scale.cfg) that reset every
layer -- there is no cross-layer persistence and no unified-address
placement decision anywhere in the unmodified engine (confirmed while
building cosma_resident_buffers.py: there's no hook for "already loaded
by a previous layer"). Drawing baseline tensors at fabricated SPM
addresses would misrepresent what it actually does. Instead the baseline
panel is a structural rendering, derived purely from producer_layer/
consumer_layers: every tensor is resident only at the instant it's
produced, and is independently re-fetched fresh at every consuming
timestep (never cached between uses, even adjacent ones).

Run from cosma/ (no PYTHONPATH needed -- unlike run_cosma.py/baseline.py,
this file never imports anything under scalesim/; model_resolver doesn't
either, so --model-json accepts a raw .tflite too, auto-exported/cached
under cosma/_exported/ the same way run_cosma.py/run_experiments.py do):
    python3 visualize_spm.py --model-json model.json --budget-kb 64
    python3 visualize_spm.py --model-json model.json --bounds-only
    python3 visualize_spm.py --model-json some_model.tflite --bounds-only

No currently-exported real model (see cosma/ITERATION_HISTORY.md) ever
triggers a nonzero spill/retrieve -- their structural minimum (M_R) and
MPMF ceiling are exactly equal. cosma/toy_spill_model.json is a small
synthetic fixture (documented as such, not a valid baseline.py/
topology_builder.py input) built specifically to demonstrate a genuine
spill/retrieve in this diagram.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from helpers import cosma_Ilp
from helpers import graph_builder
from helpers import model_resolver
from helpers import spm_allocator

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_JSON = os.path.join(HERE, 'model.json')
DEFAULT_OUT_DIR = os.path.join(HERE, 'spm_plots')

# Tensor block fill is a visual-separation aid between adjacent blocks,
# not a per-tensor legend -- a 60-130 tensor model can't give each one a
# legended identity color no matter the colormap. print_tensor_table() is
# the real source of per-tensor identity/values.
_TENSOR_PALETTE = [
    '#4c78a8', '#54a24b', '#72b7b2', '#b279a2',
    '#9d755d', '#5b8c85', '#7a6fa6', '#3d6b8f',
]
_SPILL_COLOR = '#d03b3b'      # status: critical -- an eviction happened
_RETRIEVE_COLOR = '#fab219'   # status: warning -- a DRAM round trip happened
_GRID_COLOR = '#e1e0d9'
_SPINE_COLOR = '#c3c2b7'
_TICK_COLOR = '#898781'
_LABEL_COLOR = '#57564f'


def compute_baseline_resident_action(nodes, tensors) -> dict:
    """
    Structural model of the no-COSMA regime -- SCALE-Sim's default
    scratchpad has no cross-layer persistence at all (confirmed while
    building cosma_resident_buffers.py), so a tensor is resident only at
    its own producer timestep, and is independently re-fetched from DRAM
    at *every* consumer timestep, never cached between uses. Purely
    derived from the graph -- no SCALE-Sim call, same cost as the ILP
    path. Same (tensor_id, t) -> action shape as
    cosma_Ilp.extract_results()['resident_action'], but only ever 'C' or
    'R' -- baseline never has a reason to preserve, and never chooses to
    spill because it never held onto anything to begin with.
    """
    action = {}
    for a, t in tensors.items():
        action[(a, t.producer_layer)] = 'C'
        for c in t.consumer_layers:
            action[(a, c)] = 'R'
    return action


def solve_ilp_only(nodes, tensors, memory_budget_bytes: int, ilp_time_limit_sec: float = None,
                    free_schedule: bool = False, solver: str = 'cbc') -> dict:
    """
    Steps 1-2,5-7 of run_cosma.run_cosma() only (graph + ILP) -- no
    baseline.py, no SCALE-Sim. On infeasibility or a non-optimal solve,
    bakes compute_structural_minimum_bytes()/compute_mpmf_bytes() into
    the raised error so a failing run immediately reports the right
    budget range instead of a bare pulp status string.

    free_schedule: see cosma_Ilp.build_cosma_model()'s docstring -- lets
        the ILP choose the operator order itself instead of fixing it to
        model.json's. Off by default; expect a slower solve when enabled.

    ilp_time_limit_sec: None (default) means unbounded -- matches
        run_cosma.py's own default (see its own --time-limit docstring,
        item 29 in ITERATION_HISTORY.md: defaulting to a bounded time
        limit was removed there specifically because CBC/PuLP's status
        reporting under a time limit is not reliable -- a solve that
        merely didn't finish in time can come back labeled 'Infeasible'
        (not just 'Not Solved'), which is indistinguishable from a
        genuine proof of infeasibility unless you already know better.
        Confirmed directly: a real model's free-schedule M_P solve
        (cosma_Ilp.compute_true_mpmf_bytes(), a much harder problem than
        this fixed-schedule one -- see its own docstring) reported
        'Infeasible' under a time limit despite the graph's own execution
        order being a hand-verified, zero-violation feasible solution.
        Pass an explicit value only if you're prepared for a false
        'Infeasible'/'Not Solved' on a large model.
    """
    def _bounds_message() -> str:
        floor_b, floor_t = cosma_Ilp.compute_structural_minimum_bytes(nodes, tensors)
        ceil_b, ceil_t = cosma_Ilp.compute_mpmf_bytes(nodes, tensors)
        return (f"Structural minimum (M_R): {floor_b} bytes ({floor_b / 1024:.2f} KB) "
                f"at t={floor_t} ({nodes[floor_t].op}).\n"
                f"MPMF ceiling (0 spill needed at/above): {ceil_b} bytes "
                f"({ceil_b / 1024:.2f} KB) at t={ceil_t} ({nodes[ceil_t].op}).")

    try:
        cosma_Ilp.assert_tensors_fit_budget(tensors, memory_budget_bytes)
    except AssertionError as e:
        raise RuntimeError(f"{e}\n{_bounds_message()}") from e

    prob, variables, T, A = cosma_Ilp.build_cosma_model(
        nodes, tensors, memory_budget_bytes, free_schedule=free_schedule)
    status, has_feasible_incumbent = cosma_Ilp.solve(
        prob, time_limit_sec=ilp_time_limit_sec, solver=solver)
    if status != 'Optimal' and not has_feasible_incumbent:
        caveat = (
            f" -- a time limit ({ilp_time_limit_sec}s) was set, so this status is "
            f"NOT necessarily a proof: CBC/PuLP can report 'Infeasible' (not just "
            f"'Not Solved') for a solve that simply didn't finish in time -- retry "
            f"with a longer --time-limit or omit it for an unbounded solve before "
            f"trusting this as a real infeasibility."
            if ilp_time_limit_sec and solver == 'cbc' else ""
        )
        raise RuntimeError(f"COSMA ILP did not solve to optimality: status={status}{caveat}\n"
                            f"{_bounds_message()}")
    if status != 'Optimal':
        print(f"WARNING: COSMA ILP did not prove optimality (status={status}) -- "
              f"accepting Gurobi's best incumbent found so far instead of a "
              f"proven-optimal solution.")
    return cosma_Ilp.extract_results(variables, T, A, tensors)


def spm_plan_to_runs(spm_plan):
    """
    tensor_id -> [(t_start, t_end_inclusive, address), ...]. A new run
    starts whenever the next timestep isn't exactly prev+1, or the
    address differs from the current run's -- both are checked, though
    the ILP's own constraints (Eq.1/3/4/11) guarantee a spill (which
    leaves no spm_plan entry, since S alone means C/P/R are all 0 per
    Eq.1) always precedes any address change for the same tensor, so
    gaps and address-changes always coincide in practice.
    """
    by_tensor = {}
    for (a, t) in spm_plan:
        by_tensor.setdefault(a, []).append(t)

    runs = {}
    for a, ts in by_tensor.items():
        ts.sort()
        tensor_runs = []
        run_start = run_addr = prev_t = None
        for t in ts:
            addr = spm_plan[(a, t)]
            if run_start is None:
                run_start, run_addr, prev_t = t, addr, t
            elif t == prev_t + 1 and addr == run_addr:
                prev_t = t
            else:
                tensor_runs.append((run_start, prev_t, run_addr))
                run_start, run_addr, prev_t = t, addr, t
        tensor_runs.append((run_start, prev_t, run_addr))
        runs[a] = tensor_runs
    return runs


def _style_axes(ax) -> None:
    ax.grid(True, color=_GRID_COLOR, linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_COLOR)
    ax.tick_params(colors=_TICK_COLOR, labelsize=7)


def baseline_timestep_stacks(nodes, tensors, baseline_action):
    """
    For each timestep, stack that timestep's active tensors (whatever
    baseline_action says is 'C' or 'R' there -- exactly the current
    layer's own inputs+outputs) from address 0 upward, sorted by tensor
    id for determinism. Returns (stacks, peak_bytes) where stacks[t] is
    [(tensor_id, offset, size), ...]. This is a visualization convenience
    (baseline never actually shares one address space -- see the panel's
    own subtitle), but the *set* of tensors active at t and their total
    bytes is real: it's exactly what compute_structural_minimum_bytes()
    computes per node, since baseline's peak simultaneous need at any one
    operator is the same structural quantity that bounds COSMA too --
    both are bound by the same per-operator minimum, they just differ in
    whether the allocator reuses space *across* operators.
    """
    by_timestep = {}
    for (a, t) in baseline_action:
        by_timestep.setdefault(t, []).append(a)

    stacks = {}
    peak_bytes = 0
    for t, tensor_ids in by_timestep.items():
        offset = 0
        entries = []
        for a in sorted(tensor_ids):
            size = tensors[a].size_bytes
            entries.append((a, offset, size))
            offset += size
        stacks[t] = entries
        peak_bytes = max(peak_bytes, offset)
    return stacks, peak_bytes


def _render_baseline_panel(ax, nodes, tensors, baseline_action, ylim_bytes: int) -> None:
    """Same visual grammar as the COSMA panel -- colored rectangles sized by
    real byte count, on the same byte-address y-axis -- so the two panels
    are directly comparable. The honest difference: a baseline rectangle
    spans exactly one timestep and is always restacked from address 0,
    since nothing carries over to the next timestep (no real placement is
    ever chosen -- see the subtitle); a COSMA rectangle can span many
    contiguous timesteps at a stable address because it's actually kept
    resident."""
    stacks, _ = baseline_timestep_stacks(nodes, tensors, baseline_action)
    last_pos = {}  # tensor_id -> (t, y_center), to draw the connecting line

    for t in sorted(stacks.keys()):
        for (a, offset, size) in stacks[t]:
            color = _TENSOR_PALETTE[a % len(_TENSOR_PALETTE)]
            rect = patches.Rectangle((t, offset), 1, size, facecolor=color,
                                      alpha=0.65, edgecolor='#3a3a38', linewidth=0.6, zorder=2)
            ax.add_patch(rect)
            y_center = offset + size / 2
            if baseline_action[(a, t)] == 'R':
                # Same (t, addr + size/2) convention as the COSMA panel's own
                # retrieve marker (_render_cosma_panel below), so a marker at
                # the same x/y logic means the same thing in both panels.
                ax.plot(t, y_center, marker='^', markersize=7, markerfacecolor=_RETRIEVE_COLOR,
                        markeredgecolor='white', markeredgewidth=1.0, zorder=4)
            if a in last_pos:
                pt, py = last_pos[a]
                ax.plot([pt, t], [py, y_center], linestyle=':',
                        color=color, linewidth=1.0, zorder=1)
            last_pos[a] = (t, y_center)

    ax.set_ylim(0, ylim_bytes)
    ax.set_ylabel('bytes active this timestep\n(stacked from 0 -- visual scale only)', fontsize=7)
    ax.set_title('Baseline (no cross-layer residency) -- no placement: SCALE-Sim never '
                 'packs tensors into a shared address space', fontsize=9, color=_LABEL_COLOR)
    _style_axes(ax)


def _render_cosma_panel(ax, nodes, tensors, result, memory_budget_bytes: int, ylim_bytes: int,
                         compacted: bool = False) -> None:
    spm_plan = result['spm_plan']
    resident_action = result['resident_action']
    runs_by_tensor = spm_plan_to_runs(spm_plan)

    for a, runs in runs_by_tensor.items():
        color = _TENSOR_PALETTE[a % len(_TENSOR_PALETTE)]
        size = tensors[a].size_bytes
        for (t_start, t_end, addr) in runs:
            width = t_end - t_start + 1
            rect = patches.Rectangle((t_start, addr), width, size, facecolor=color,
                                      alpha=0.65, edgecolor='#3a3a38', linewidth=0.6, zorder=2)
            ax.add_patch(rect)
            if size >= memory_budget_bytes * 0.03:
                ax.text(t_start + width / 2, addr + size / 2, f"t{a}\n@{addr}",
                        ha='center', va='center', fontsize=6, color='white', zorder=3)

    for (a, t), action in resident_action.items():
        size = tensors[a].size_bytes
        if action == 'S':
            addr = spm_plan[(a, t - 1)]
            ax.plot(t, addr + size / 2, marker='v', markersize=8, markerfacecolor=_SPILL_COLOR,
                    markeredgecolor='white', markeredgewidth=1.2, zorder=4)
        elif action == 'R':
            addr = spm_plan[(a, t)]
            ax.plot(t, addr + size / 2, marker='^', markersize=8, markerfacecolor=_RETRIEVE_COLOR,
                    markeredgecolor='white', markeredgewidth=1.2, zorder=4)

    ax.set_xlabel('timestep (layer)')
    ax.set_ylabel('SPM address (bytes)', fontsize=8)
    ax.set_ylim(0, ylim_bytes)
    subtitle = (" -- repacked toward 0 for readability, same plan (see "
                "compact_spm_plan())" if compacted else
                " -- solver's own raw addresses (gaps expected, not a bug "
                "-- see compact_spm_plan())")
    ax.set_title(f"COSMA plan @ {memory_budget_bytes / 1024:.2f} KB budget{subtitle}",
                 fontsize=8, color=_LABEL_COLOR)
    _style_axes(ax)


def render_comparison(nodes, tensors, baseline_action, result, memory_budget_bytes: int,
                       out_path: str, title: str = None, compact: bool = True) -> None:
    """Two stacked panels, one combined PNG -- the deliverable ('visualization
    for both plans'), not two separate files. Never calls plt.show().

    compact: if True (default), the COSMA panel is rendered from a
    repacked-toward-0 equivalent of result['spm_plan'] (see
    spm_allocator.compact_spm_plan()) instead of the solver's own literal
    addresses -- same plan, same resident_action, purely a readability
    improvement (COSMA's ILP has no preference at all for compact
    placement, so the raw addresses tend to scatter with pointless gaps;
    see compact_spm_plan()'s own docstring). Falls back to the raw
    addresses (with a printed warning, not a crash) if compaction ever
    fails -- dynamic storage allocation with variable-size objects is
    NP-hard in general, so this heuristic is not guaranteed to always
    succeed, though it has on every model tried so far. Pass False to
    always see the solver's own literal addresses (e.g. to sanity-check
    the ILP's own placement choices, not the repacked view).
    """
    max_t = max(nodes.keys())
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(max(10, max_t * 0.35), 10),
        gridspec_kw={'height_ratios': [1, 1]}, sharex=True)

    # Both panels share one byte-address y-axis so bar heights are directly
    # comparable -- sized to fit whichever is larger: the COSMA budget being
    # tested, or baseline's own peak simultaneous requirement (which is the
    # same per-operator minimum, M_R, that bounds COSMA too; see
    # baseline_timestep_stacks()'s docstring).
    _, baseline_peak_bytes = baseline_timestep_stacks(nodes, tensors, baseline_action)
    ylim_bytes = max(memory_budget_bytes, baseline_peak_bytes)

    cosma_result = result
    compacted = False
    if compact:
        try:
            compact_plan = spm_allocator.compact_spm_plan(
                tensors, result['resident_action'], memory_budget_bytes)
            cosma_result = dict(result, spm_plan=compact_plan)
            compacted = True
        except spm_allocator.SpmAllocationError as e:
            print(f"[visualize_spm] compaction failed, falling back to the "
                  f"solver's own raw addresses: {e}")

    _render_baseline_panel(ax_top, nodes, tensors, baseline_action, ylim_bytes)
    _render_cosma_panel(ax_bottom, nodes, tensors, cosma_result, memory_budget_bytes, ylim_bytes,
                         compacted=compacted)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_bottom.set_xlim(-0.5, max_t + 1.5)
    tick_step = max(1, max_t // 20)
    ticks = list(range(0, max_t + 1, tick_step))
    ax_bottom.set_xticks(ticks)
    # schedule_layer_at_t maps abstract timestep -> real layer id (identity
    # under a fixed schedule, a real reordering under free_schedule=True --
    # see cosma_Ilp.extract_results()) -- ticks must be labeled by whichever
    # layer actually runs at t, not by t itself.
    schedule_layer_at_t = result.get('schedule_layer_at_t', {t: t for t in nodes})
    ax_bottom.set_xticklabels(
        [f"{t}\n{nodes[schedule_layer_at_t[t]].op}" for t in ticks], fontsize=6)

    legend_handles = [
        Line2D([], [], marker='v', linestyle='', markerfacecolor=_SPILL_COLOR,
               markeredgecolor='white', markersize=8, label='Spill'),
        Line2D([], [], marker='^', linestyle='', markerfacecolor=_RETRIEVE_COLOR,
               markeredgecolor='white', markersize=8, label='DRAM fetch (retrieve / baseline reload)'),
    ]
    fig.legend(handles=legend_handles, loc='upper right', fontsize=7, frameon=False)

    fig.suptitle(title or 'SPM occupancy: baseline vs. COSMA', fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_tensor_table(tensors, baseline_action: dict, result: dict) -> None:
    """Plain-text per-tensor event listing for both regimes -- the text-twin
    of the comparison figure, since color can't carry full identity at scale."""
    spm_plan = result['spm_plan']
    resident_action = result['resident_action']

    print(f"\n{'tensor':>8}  {'size(B)':>8}  {'baseline events':<30}  cosma events")
    for a in sorted(tensors.keys()):
        size = tensors[a].size_bytes
        base_events = sorted((t, act) for (aa, t), act in baseline_action.items() if aa == a)
        base_str = ', '.join(f"{act}@{t}" for t, act in base_events)

        cosma_events = sorted((t, act) for (aa, t), act in resident_action.items() if aa == a)
        cosma_parts = []
        for t, act in cosma_events:
            if act in ('C', 'P', 'R'):
                cosma_parts.append(f"{act}@{t}(addr={spm_plan.get((a, t))})")
            else:
                cosma_parts.append(f"{act}@{t}")
        cosma_str = ', '.join(cosma_parts)

        print(f"{a:>8}  {size:>8}  {base_str:<30}  {cosma_str}")


def default_out_path(model_json_path: str, memory_budget_bytes: int,
                      array_tag: str = None, schedule_tag: str = None) -> str:
    """
    cosma/spm_plots/<model>[_<array_tag>]_<budget>kb[_<schedule_tag>].png

    No timestamp by design -- re-running the same (model, array config,
    budget, schedule mode) overwrites its previous plot rather than
    accumulating one file per run; pass a distinct budget/config/schedule
    (which this naming already disambiguates) or --out/--plot-out to keep
    an old one around.

    array_tag: e.g. an array-size fingerprint like '64x64', so re-running
        under a different SCALE-Sim config doesn't overwrite the previous
        plot. Left as a caller-supplied string, not read from a config
        here, since this module deliberately never imports anything under
        scalesim/ (see the module docstring) -- callers that already have
        scalesim access (run_cosma.py, run_experiments.py) compute it
        themselves; visualize_spm.py's own CLI omits it (no scalesim
        access), so its plots are untagged by array size.
    schedule_tag: 'S' (static -- model.json's fixed order) or 'D' (dynamic
        -- free_schedule=True), so a static and a rescheduled run of the
        same (model, budget) don't overwrite each other.
    """
    parent = os.path.basename(os.path.dirname(os.path.abspath(model_json_path)))
    stem = os.path.splitext(os.path.basename(model_json_path))[0]
    name = parent if parent and parent != 'cosma' else stem

    budget_kb_str = f"{memory_budget_bytes / 1024:.3f}".rstrip('0').rstrip('.').replace('.', 'p')
    parts = [name]
    if array_tag:
        parts.append(array_tag)
    parts.append(f"{budget_kb_str}kb")
    if schedule_tag:
        parts.append(schedule_tag)
    return os.path.join(DEFAULT_OUT_DIR, "_".join(parts) + ".png")


def print_budget_bounds(nodes, tensors) -> dict:
    """
    --bounds-only output: prints M_R/MPMF in KB with each one's argmax op,
    and whether an interesting spill/retrieve range exists at all. Also
    usable as a library call, not just CLI output.
    """
    floor_b, floor_t = cosma_Ilp.compute_structural_minimum_bytes(nodes, tensors)
    ceil_b, ceil_t = cosma_Ilp.compute_mpmf_bytes(nodes, tensors)

    print(f"Structural minimum (M_R):        {floor_b:>10d} bytes ({floor_b / 1024:8.2f} KB) "
          f"at t={floor_t} ({nodes[floor_t].op if floor_t is not None else '-'})")
    print(f"MPMF ceiling (0 spill at/above): {ceil_b:>10d} bytes ({ceil_b / 1024:8.2f} KB) "
          f"at t={ceil_t} ({nodes[ceil_t].op if ceil_t is not None else '-'})")
    print("(MPMF ceiling above is the fixed-schedule proxy, not the paper's "
          "true M_P -- pass --true-mpmf for the real thing.)")
    if ceil_b > floor_b:
        print(f"Interesting spill/retrieve range: ({floor_b / 1024:.2f}, {ceil_b / 1024:.2f}) KB")
    else:
        print("No interesting range -- floor == ceiling, spill/retrieve can never be "
              "nonzero for this model at any feasible budget.")

    return {'structural_minimum_bytes': floor_b, 'structural_minimum_t': floor_t,
            'mpmf_bytes': ceil_b, 'mpmf_t': ceil_t}


def print_true_mpmf(nodes, tensors, m_r_bytes: int, time_limit_sec: float = None,
                     solver: str = 'cbc') -> dict:
    """
    The paper's real M_P (§III-E1/Eq.13-15, cosma_Ilp.compute_true_mpmf_bytes()
    -- an actual free-schedule ILP solve, not print_budget_bounds()'s instant
    fixed-schedule proxy), plus the derived M_H = (M_R + M_P) / 2. Opt-in
    (--true-mpmf) since this pays for a real solve -- can be slow on a large
    model, unlike everything else --bounds-only prints. m_r_bytes is passed
    in (from print_budget_bounds()'s own already-computed M_R) rather than
    recomputed, since M_H needs it.
    """
    true_mp_bytes, schedule = cosma_Ilp.compute_true_mpmf_bytes(
        nodes, tensors, time_limit_sec=time_limit_sec, solver=solver)
    m_h_bytes = (m_r_bytes + true_mp_bytes) / 2

    print(f"True M_P (real ILP solve, §III-E1/Eq.13-15): {true_mp_bytes:>10d} bytes "
          f"({true_mp_bytes / 1024:8.2f} KB)")
    print(f"M_H = (M_R + M_P) / 2:                        {m_h_bytes:>13.1f} bytes "
          f"({m_h_bytes / 1024:8.2f} KB)")
    return {'true_mpmf_bytes': true_mp_bytes, 'm_h_bytes': m_h_bytes, 'schedule': schedule}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-json', default=DEFAULT_MODEL_JSON,
                         help='Path to a model.json, or a raw .tflite to auto-export '
                              '(default: cosma/model.json).')
    parser.add_argument('--exporter', default=model_resolver.DEFAULT_EXPORTER,
                         help='Path to trim/python_scripts/export_model.py, used only '
                              'when --model-json is a .tflite.')
    parser.add_argument('--export-dir', default=model_resolver.DEFAULT_EXPORT_DIR,
                         help='Where auto-exported model.json files are cached '
                              '(default: cosma/_exported).')
    parser.add_argument('--force-export', action='store_true',
                         help='Re-export even if a cached model.json already exists.')
    parser.add_argument('--budget-kb', type=float, default=128,
                         help='SPM budget in KB for the COSMA panel (default: 128). '
                              'Ignored with --bounds-only.')
    parser.add_argument('--bounds-only', action='store_true',
                         help='Print the M_R floor / MPMF ceiling and exit -- no ILP '
                              'solve, no plot, no SCALE-Sim.')
    parser.add_argument('--true-mpmf', action='store_true',
                         help='Print M_R, the paper\'s real M_P (a free-schedule ILP '
                              'solve, not the instant fixed-schedule MPMF proxy), and '
                              'derived M_H, then exit -- no plot, no SCALE-Sim. Implies '
                              '--bounds-only. Not instant -- pays for a real ILP solve, '
                              'can be slow on a large model.')
    parser.add_argument('--out', default=None,
                         help='Output PNG path (default: cosma/spm_plots/<model>_<budget>kb.png).')
    parser.add_argument('--time-limit', type=float, default=None,
                         help='CBC solve time limit, seconds. Default: unbounded -- run '
                              'until CBC proves Optimal or Infeasible, however long that '
                              'takes. Matches run_cosma.py\'s own default; a bounded time '
                              'limit here can produce a false "Infeasible" for a solve '
                              'that simply ran out of time (see solve_ilp_only()\'s '
                              'docstring) -- only set this if you\'ve confirmed that '
                              'risk is acceptable for your model.')
    parser.add_argument('--free-schedule', action='store_true',
                         help="Let the ILP choose the operator schedule itself instead of "
                              "fixing it to model.json's own order -- see "
                              "cosma_Ilp.build_cosma_model()'s free_schedule docstring. "
                              "Ignored with --bounds-only.")
    parser.add_argument('--raw-addresses', action='store_true',
                         help="Show the solver's own literal SPM addresses in the COSMA "
                              "panel instead of the default repacked-toward-0 view -- "
                              "nothing in COSMA's ILP rewards compact placement, so the "
                              "raw addresses tend to scatter with pointless gaps (see "
                              "spm_allocator.compact_spm_plan()). Ignored with --bounds-only.")
    parser.add_argument('--solver', choices=['cbc', 'gurobi'], default='gurobi',
                         help="ILP solver backend (default: cbc, no license needed). "
                              "'gurobi' requires a working Gurobi license -- see "
                              "cosma_Ilp.solve()'s docstring -- but measured ~600x faster "
                              "than CBC on Inception-V3-sized problems in this project's "
                              "own profiling; worth using whenever available, especially "
                              "with --true-mpmf/--free-schedule on a large model.")
    args = parser.parse_args()

    model_json_path = model_resolver.resolve_model_json(
        args.model_json, args.exporter, args.export_dir, args.force_export)
    nodes, tensors = graph_builder.load_graph(model_json_path)

    if args.bounds_only or args.true_mpmf:
        bounds = print_budget_bounds(nodes, tensors)
        if args.true_mpmf:
            print_true_mpmf(nodes, tensors, bounds['structural_minimum_bytes'],
                             time_limit_sec=args.time_limit, solver=args.solver)
        return

    memory_budget_bytes = int(args.budget_kb * 1024)
    baseline_action = compute_baseline_resident_action(nodes, tensors)
    result = solve_ilp_only(nodes, tensors, memory_budget_bytes, args.time_limit,
                             free_schedule=args.free_schedule, solver=args.solver)

    out_path = args.out or default_out_path(
        model_json_path, memory_budget_bytes,
        schedule_tag='D' if args.free_schedule else 'S')
    render_comparison(nodes, tensors, baseline_action, result, memory_budget_bytes, out_path,
                       title=f"{os.path.basename(model_json_path)} -- SPM occupancy: "
                             f"baseline vs. COSMA",
                       compact=not args.raw_addresses)
    print_tensor_table(tensors, baseline_action, result)
    print(f"\nSaved SPM occupancy comparison to {out_path}")


if __name__ == '__main__':
    main()
