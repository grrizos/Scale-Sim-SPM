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
  helpers/model_resolver.resolve_model_json()
  helpers/graph_builder.load_graph()
  helpers/baseline.run_baseline() / run_cosma_aware()
  helpers/cosma_Ilp.assert_tensors_fit_budget() / compute_true_mpmf_bytes()
  run_cosma.run_cosma()  -- optional 5th "cosma_native" comparison row
"""
import argparse
import csv
import functools
import os

from helpers import graph_builder
from helpers import baseline
from helpers import cosma_Ilp
from helpers import model_resolver
from helpers import schedule_variants
from helpers import replacement_engine
from helpers import belady_policy
from helpers import ilp_greedy_policy
from helpers import tflite_arena_allocator
import run_cosma

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), 'configs', 'scale.cfg')

CONV_LIKE_OPS = ('CONV2D', 'DEPTHWISE_CONV2D')

POLICY_FNS = {
    'belady': belady_policy.choose_victims,
    'ilp_greedy': ilp_greedy_policy.choose_victims,
}


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
                        verbose: bool = True) -> dict:
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
        schedule=schedule, verbose=False)

    accounting = _account(nodes, tensors, resident_action, schedule,
                           layer_stats, plan_stats, bandwidth_bytes_per_cycle)
    summary = {'status': 'Optimal', 'memory_budget_bytes': memory_budget_bytes,
               'spm_plan': spm_plan, 'resident_action': resident_action}
    summary.update(accounting)
    if verbose:
        print(f"  non-compulsory bytes: {summary['total_non_compulsory_access_bytes']}, "
              f"DRAM reduction: {summary['dram_traffic_reduction_pct']:.1f}%, "
              f"speedup: {summary['speedup']:.4f}x")
    return summary


def run_all_paper_baselines(model_json_path: str, config_path: str = DEFAULT_CONFIG,
                             memory_budget_bytes: int = 128 * 1024,
                             include_cosma_native: bool = True,
                             mpmf_time_limit_sec: float = None,
                             ilp_greedy_time_limit_sec: float = 10,
                             solver: str = 'cbc',
                             exporter: str = model_resolver.DEFAULT_EXPORTER,
                             export_dir: str = model_resolver.DEFAULT_EXPORT_DIR,
                             force_export: bool = False,
                             verbose: bool = True) -> dict:
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

    Returns {'default+belady', 'default+ilp_greedy', 'mpmf+belady',
    'mpmf+ilp_greedy'[, 'cosma_native']: summary dict (or
    {'status': 'ERROR', 'error': str} on failure)}.
    """
    model_json_path = model_resolver.resolve_model_json(
        model_json_path, exporter, export_dir, force_export)
    nodes, tensors = graph_builder.load_graph(model_json_path)
    cosma_Ilp.assert_tensors_fit_budget(tensors, memory_budget_bytes)

    bandwidth_bytes_per_cycle = _default_bandwidth_words_per_cycle(config_path)

    if verbose:
        print(f"Running SCALE-Sim baseline for {model_json_path} ...")
    layer_stats = baseline.run_baseline(model_json_path, config_path)

    default_sched = schedule_variants.default_operator_schedule(nodes)
    if verbose:
        print("Solving MPMF schedule ILP (cosma_Ilp.compute_true_mpmf_bytes) ...")
    _, mpmf_tensor_sched = cosma_Ilp.compute_true_mpmf_bytes(
        nodes, tensors, time_limit_sec=mpmf_time_limit_sec, solver=solver)
    mpmf_sched = schedule_variants.mpmf_operator_schedule(nodes, tensors, mpmf_tensor_sched)

    schedules = {'default': default_sched, 'mpmf': mpmf_sched}
    results = {}
    for sched_name, sched in schedules.items():
        for policy_name, policy_fn in POLICY_FNS.items():
            combo = f"{sched_name}+{policy_name}"
            if verbose:
                print(f"\n=== {combo} ===")
            try:
                results[combo] = run_paper_baseline(
                    nodes, tensors, model_json_path, config_path, sched,
                    memory_budget_bytes, policy_fn, layer_stats,
                    bandwidth_bytes_per_cycle,
                    ilp_greedy_time_limit_sec=ilp_greedy_time_limit_sec,
                    ilp_greedy_solver=solver, verbose=verbose)
            except (tflite_arena_allocator.TfliteArenaAllocationError,
                    replacement_engine.ReplacementInfeasible) as e:
                results[combo] = {'status': 'ERROR', 'error': f"{type(e).__name__}: {e}"}
                if verbose:
                    print(f"  FAILED: {type(e).__name__}: {e}")

    if include_cosma_native:
        if verbose:
            print("\n=== cosma_native ===")
        try:
            results['cosma_native'] = run_cosma.run_cosma(
                model_json_path=model_json_path, config_path=config_path,
                memory_budget_bytes=memory_budget_bytes, layer_stats=layer_stats,
                bandwidth_bytes_per_cycle=bandwidth_bytes_per_cycle,
                verbose=False, save_plot=False, solver=solver)
            if verbose:
                r = results['cosma_native']
                print(f"  non-compulsory bytes: {r['total_non_compulsory_access_bytes']}, "
                      f"DRAM reduction: {r['dram_traffic_reduction_pct']:.1f}%, "
                      f"speedup: {r['speedup']:.4f}x")
        except Exception as e:  # noqa: BLE001 -- one failed combination shouldn't abort the rest
            results['cosma_native'] = {'status': 'ERROR', 'error': f"{type(e).__name__}: {e}"}
            if verbose:
                print(f"  FAILED: {type(e).__name__}: {e}")

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
    parser.add_argument('--time-limit', type=float, default=None,
                         help='Time limit (seconds) for the MPMF schedule ILP solve. '
                              'Default: unbounded.')
    parser.add_argument('--ilp-greedy-time-limit', type=float, default=10,
                         help='Per-decision-point time limit (seconds) for the local '
                              'ILP-greedy replacement policy solve. Default: 10.')
    parser.add_argument('--solver', choices=['cbc', 'gurobi'], default='gurobi',
                         help='ILP solver for the MPMF schedule and ILP-greedy replacement '
                              '(default: cbc, no license needed). See cosma_Ilp.solve().')
    parser.add_argument('--no-cosma-native', action='store_true',
                         help="Skip the COSMA-native comparison row (run_cosma.run_cosma()).")
    parser.add_argument('--out-csv', default=None,
                         help='Path to write a combined results CSV across all budgets.')
    args = parser.parse_args()

    all_rows = []
    for budget_kb in args.budgets_kb:
        results = run_all_paper_baselines(
            model_json_path=args.model_json, config_path=args.config,
            memory_budget_bytes=int(budget_kb * 1024),
            include_cosma_native=not args.no_cosma_native,
            mpmf_time_limit_sec=args.time_limit,
            ilp_greedy_time_limit_sec=args.ilp_greedy_time_limit,
            solver=args.solver, exporter=args.exporter, export_dir=args.export_dir,
            force_export=args.force_export, verbose=True)
        print_comparison_table(results, budget_kb)
        for combo, r in results.items():
            all_rows.append({
                'model': args.model_json, 'budget_kb': budget_kb, 'combo': combo,
                'status': r.get('status', 'Optimal'),
                'total_non_compulsory_access_bytes': r.get('total_non_compulsory_access_bytes', ''),
                'dram_traffic_reduction_pct': r.get('dram_traffic_reduction_pct', ''),
                'speedup': r.get('speedup', ''),
                'error': r.get('error', ''),
            })

    if args.out_csv:
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'model', 'budget_kb', 'combo', 'status',
                'total_non_compulsory_access_bytes', 'dram_traffic_reduction_pct',
                'speedup', 'error'])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} rows to {args.out_csv}")
