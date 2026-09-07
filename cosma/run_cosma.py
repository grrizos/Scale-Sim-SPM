# cosma/run_cosma.py
"""
Orchestrates the full COSMA + SCALE-Sim pipeline:
  1. graph_builder.load_graph()  -> nodes, tensors
  2. cosma_Ilp.build_cosma_model()/solve()/extract_results()
     -> ordered_layers (fixed, see cosma_Ilp docstring), resident_action
     (per tensor/timestep: C/P/R/S), spm_plan
  3. baseline.run_baseline()     -> the plain, COSMA-unaware SCALE-Sim
     simulation (no cross-layer memory sharing at all -- "the original
     boxes"), independent of any COSMA plan or budget.
     baseline.run_cosma_aware() -> the SAME simulation, but with
     resident_action installed via CosmaResidentReadBuffer/
     CosmaResidentWriteBuffer (scalesim/memory/cosma_resident_buffers.py)
     so a 'P'-resident ifmap fetch and every ofmap write are genuinely
     simulated as free, not asserted to be after the fact. Depends on the
     budget (indirectly, via resident_action) so this one must be re-run
     per budget.
  4. Combine: total_cycles = sum_t max(compute_cycles[t], dram[t] / BW).

Why two separate SCALE-Sim passes instead of one plus analytic
adjustment: earlier versions of this file computed a single SCALE-Sim
baseline and then zeroed out / substituted specific numbers in Python
based on resident_action, because SCALE-Sim's engine was believed
off-limits to modify. That's no longer the constraint -- see
scalesim/memory/cosma_resident_buffers.py and
cosma/ITERATION_HISTORY.md for the investigation and design. The two
buffer classes there change behavior *only* in the one case COSMA's ILP
has already resolved (a tensor asserted resident needs no fetch; a
freshly-created tensor that stays resident needs no drain-to-DRAM) --
every other code path is byte-identical to SCALE-Sim's unmodified
behavior (regression-tested in cosma/ITERATION_HISTORY.md).

What's still idealized, and why it has to be: a tensor's 'S' (spilled)
event -- evicting an already-resident tensor, unrelated to any layer's
own input/output -- has no SCALE-Sim analog at all. SCALE-Sim only ever
simulates "this layer's own input fetch" and "this layer's own output
write," never "evict some unrelated tensor sitting in memory right now."
There's no operator to attach a real demand matrix to, so this component
stays COSMA's own idealized `size(a)` byte count converted to cycles via
the same bandwidth estimate used for the overall max() formula below --
disclosed, not hidden.

Filter (weight) DRAM bytes are always charged in full in both scenarios,
unconditionally -- weights are never COSMA-tracked (always fetched fresh,
per the "activation tensors only" design decision).
"""
import argparse
import os

from helpers import graph_builder
from helpers import baseline
from helpers import cosma_Ilp
from helpers import model_resolver
import visualize_spm

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_JSON = os.path.join(HERE, 'model.json')
DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), 'configs', 'scale.cfg')


def _default_bandwidth_words_per_cycle(config_path: str) -> float:
    """
    SCALE-Sim's scale.cfg here runs in CALC (estimate) bandwidth mode, which
    doesn't expose one unified DRAM bandwidth number -- it estimates
    per-buffer backing bandwidth from the array width instead (see
    single_layer_sim.run()'s `ofmap_backing_bw = arr_col` default). We reuse
    that same array-width estimate as the system DRAM bandwidth for
    converting extra_dram_bytes into cycles.
    """
    from scalesim.scale_config import scale_config
    config = scale_config()
    config.read_conf_file(config_path)
    if config.use_user_dram_bandwidth():
        return float(config.get_bandwidths_as_list()[0])
    _, arr_col = config.get_array_dims()
    return float(arr_col)


def _array_dims_tag(config_path: str) -> str:
    """'<ArrayHeight>x<ArrayWidth>' from a scale.cfg, e.g. '64x64' -- used
    to tag the default plot filename so re-running the same (model,
    budget) under a different array size doesn't silently overwrite the
    previous plot (see run_experiments.py's identical tag for logs, added
    for the same reason)."""
    from scalesim.scale_config import scale_config
    config = scale_config()
    config.read_conf_file(config_path)
    arr_row, arr_col = config.get_array_dims()
    return f"{arr_row}x{arr_col}"


def run_cosma(model_json_path: str = DEFAULT_MODEL_JSON,
              config_path: str = DEFAULT_CONFIG,
              memory_budget_bytes: int = 128 * 1024,
              bandwidth_bytes_per_cycle: float = None,
              ilp_time_limit_sec: float = 120,
              layer_stats: dict = None,
              verbose: bool = True,
              save_plot: bool = False,
              plot_out_path: str = None,
              exporter: str = model_resolver.DEFAULT_EXPORTER,
              export_dir: str = model_resolver.DEFAULT_EXPORT_DIR,
              force_export: bool = False) -> dict:
    """
    model_json_path may also be a raw .tflite file -- it's auto-exported
    to model.json and cached under export_dir (see helpers/model_resolver.py,
    the same auto-export run_experiments.py already did; factored out so
    run_cosma.py doesn't require an already-exported model.json either).

    layer_stats: optional precomputed baseline.run_baseline() output, to
        skip re-running SCALE-Sim (which doesn't depend on memory_budget_bytes
        at all, so callers sweeping several budgets for the same model
        should compute it once and pass it in -- see run_experiments.py).

    save_plot: if True, also renders visualize_spm.py's baseline-vs-COSMA
        occupancy diagram for this exact run and saves it (default path:
        visualize_spm.default_out_path() tagged with this run's array size,
        e.g. spm_plots/model_128kb_64x64.png, so a re-run at a different
        array size doesn't overwrite the previous plot; override with
        plot_out_path). Reuses the ILP solve already
        done above (cosma_Ilp.extract_results()'s `result`) instead of
        re-solving -- calling visualize_spm.py separately as a second
        command would pay for the ILP solve twice, expensive on anything
        Inception-V3-sized. Off by default since run_experiments.py calls
        run_cosma() in a tight per-budget sweep loop where a plot per call
        isn't wanted; the CLI below turns it on by default.
    """
    model_json_path = model_resolver.resolve_model_json(
        model_json_path, exporter, export_dir, force_export)

    nodes, tensors = graph_builder.load_graph(model_json_path)

    # Cheap, baseline-independent check first: fail fast on a hopeless
    # budget instead of paying for a full SCALE-Sim run only to hit this
    # same assertion afterwards inside build_cosma_model().
    cosma_Ilp.assert_tensors_fit_budget(tensors, memory_budget_bytes)

    if bandwidth_bytes_per_cycle is None:
        bandwidth_bytes_per_cycle = _default_bandwidth_words_per_cycle(config_path)

    if layer_stats is None:
        layer_stats = baseline.run_baseline(model_json_path, config_path)

    prob, variables, T, A = cosma_Ilp.build_cosma_model(
        nodes, tensors, memory_budget_bytes=memory_budget_bytes)
    status = cosma_Ilp.solve(prob, time_limit_sec=ilp_time_limit_sec)
    if status != 'Optimal':
        raise RuntimeError(f"COSMA ILP did not solve to optimality: status={status}")

    result = cosma_Ilp.extract_results(variables, T, A, tensors)
    resident_action = result['resident_action']

    # Real, engine-driven second simulation pass -- see module docstring.
    # Budget-dependent (via resident_action), so this always re-runs.
    cosma_stats = baseline.run_cosma_aware(model_json_path, config_path, resident_action)

    CONV_LIKE_OPS = ('CONV2D', 'DEPTHWISE_CONV2D')

    baseline_total_cycles = 0
    cosma_total_cycles = 0
    baseline_dram_bytes = 0
    cosma_dram_bytes = 0
    total_idealized_spill_bytes = 0
    total_idealized_retrieve_bytes = 0
    total_real_retrieve_bytes = 0
    total_ifmap_residency_credit_bytes = 0
    total_ofmap_residency_credit_bytes = 0
    layer_bound_breakdown = []  # per-layer compute-vs-memory-bound record, see below
    for t in result['ordered_layers']:
        base_s = layer_stats[t]
        cosma_s = cosma_stats[t]
        node = nodes[t]

        # A retrieved tensor only has a *real* SCALE-Sim number to draw on
        # when it's the tracked activation input of a conv-like layer --
        # that's the only case run_cosma_aware() actually simulated a
        # fetch for (see baseline.py's _simulate_layer). A retrieve
        # consumed by a non-conv layer (e.g. ADD -- SCALE-Sim never
        # simulates those at all) has no real event to draw from, exactly
        # like a spill, and needs the same idealized fallback -- missing
        # this charged a real retrieve as free.
        is_conv_like = node.op in CONV_LIKE_OPS
        real_retrieve_tensor = (
            node.activation_inputs[0]
            if is_conv_like and node.activation_inputs
               and resident_action.get((node.activation_inputs[0], t)) == 'R'
            else None
        )

        idealized_bytes = sum(
            tensors[a].size_bytes
            for a in A
            if resident_action.get((a, t)) in ('S', 'R') and a != real_retrieve_tensor
        )
        idealized_spill_bytes = sum(
            tensors[a].size_bytes for a in A if resident_action.get((a, t)) == 'S')
        total_idealized_spill_bytes += idealized_spill_bytes
        total_idealized_retrieve_bytes += idealized_bytes - idealized_spill_bytes
        if real_retrieve_tensor is not None:
            total_real_retrieve_bytes += cosma_s['ifmap_dram_bytes']

        # The dominant, usually-invisible-in-spill/retrieve-numbers effect:
        # bytes SCALE-Sim's own engine confirms are avoidable simply by
        # keeping activations on-chip across layers -- 'P' residency
        # (ifmap) and every layer's own output never needing to leave the
        # chip at creation (ofmap, unconditional -- see module docstring).
        # This is COSMA's real contribution whenever spill/retrieve are 0.
        total_ifmap_residency_credit_bytes += (
            base_s['ifmap_dram_bytes'] - cosma_s['ifmap_dram_bytes'])
        total_ofmap_residency_credit_bytes += (
            base_s['ofmap_dram_bytes'] - cosma_s['ofmap_dram_bytes'])

        baseline_compulsory = (base_s['ifmap_dram_bytes'] + base_s['ofmap_dram_bytes']
                                + base_s['filter_dram_bytes'])
        cosma_compulsory = (cosma_s['ifmap_dram_bytes'] + cosma_s['ofmap_dram_bytes']
                             + cosma_s['filter_dram_bytes'] + idealized_bytes)

        # Which side of max(compute_cycles, dram_bytes/bandwidth) actually
        # wins, per layer -- this is the number the aggregate speedup
        # figure hides. COSMA's DRAM reduction can only ever turn into a
        # real speedup on a layer where memory was the bottleneck to begin
        # with; a layer that's already compute-bound in the baseline stays
        # exactly as slow no matter how much DRAM traffic COSMA removes.
        baseline_mem_cycles = baseline_compulsory / bandwidth_bytes_per_cycle
        cosma_mem_cycles = cosma_compulsory / bandwidth_bytes_per_cycle
        layer_bound_breakdown.append({
            't': t, 'op': node.op,
            'baseline_compute_cycles': base_s['compute_cycles'],
            'baseline_mem_cycles': baseline_mem_cycles,
            'baseline_bound': 'memory' if baseline_mem_cycles > base_s['compute_cycles'] else 'compute',
            'cosma_compute_cycles': cosma_s['compute_cycles'],
            'cosma_mem_cycles': cosma_mem_cycles,
            'cosma_bound': 'memory' if cosma_mem_cycles > cosma_s['compute_cycles'] else 'compute',
        })

        baseline_total_cycles += max(base_s['compute_cycles'], baseline_mem_cycles)
        cosma_total_cycles += max(cosma_s['compute_cycles'], cosma_mem_cycles)
        baseline_dram_bytes += baseline_compulsory
        cosma_dram_bytes += cosma_compulsory

    baseline_memory_bound_layers = sum(
        1 for r in layer_bound_breakdown if r['baseline_bound'] == 'memory')
    cosma_memory_bound_layers = sum(
        1 for r in layer_bound_breakdown if r['cosma_bound'] == 'memory')

    dram_traffic_reduction_pct = (
        100.0 * (baseline_dram_bytes - cosma_dram_bytes) / baseline_dram_bytes
        if baseline_dram_bytes > 0 else 0.0
    )
    speedup = (baseline_total_cycles / cosma_total_cycles
               if cosma_total_cycles > 0 else float('inf'))

    plot_path = None
    if save_plot:
        baseline_action = visualize_spm.compute_baseline_resident_action(nodes, tensors)
        plot_path = plot_out_path or visualize_spm.default_out_path(
            model_json_path, memory_budget_bytes, tag=_array_dims_tag(config_path))
        visualize_spm.render_comparison(
            nodes, tensors, baseline_action, result, memory_budget_bytes, plot_path,
            title=f"{os.path.basename(model_json_path)} -- SPM occupancy: "
                  f"baseline vs. COSMA @ {memory_budget_bytes / 1024:.2f} KB")

    summary = {
        'status': status,
        'memory_budget_bytes': memory_budget_bytes,
        'bandwidth_bytes_per_cycle': bandwidth_bytes_per_cycle,
        'baseline_total_cycles': baseline_total_cycles,
        'cosma_total_cycles': cosma_total_cycles,
        'baseline_dram_bytes': baseline_dram_bytes,
        'cosma_dram_bytes': cosma_dram_bytes,
        'total_idealized_spill_bytes': total_idealized_spill_bytes,
        'total_idealized_retrieve_bytes': total_idealized_retrieve_bytes,
        'total_real_retrieve_bytes': total_real_retrieve_bytes,
        'total_ifmap_residency_credit_bytes': total_ifmap_residency_credit_bytes,
        'total_ofmap_residency_credit_bytes': total_ofmap_residency_credit_bytes,
        'dram_traffic_reduction_pct': dram_traffic_reduction_pct,
        'speedup': speedup,
        'spm_plan': result['spm_plan'],
        'resident_action': resident_action,
        'layer_bound_breakdown': layer_bound_breakdown,
        'baseline_memory_bound_layers': baseline_memory_bound_layers,
        'cosma_memory_bound_layers': cosma_memory_bound_layers,
        'plot_path': plot_path,
    }

    if verbose:
        print(f"SPM budget: {memory_budget_bytes} bytes")
        print(f"ILP status: {status}")
        print(f"--- COSMA's contribution (real, SCALE-Sim-engine-simulated) ---")
        print(f"Ifmap bytes avoided by keeping activations resident ('P'): "
              f"{total_ifmap_residency_credit_bytes}")
        print(f"Ofmap bytes avoided (a layer's own output never leaves "
              f"the chip at creation): {total_ofmap_residency_credit_bytes}")
        print(f"--- COSMA's overhead to fit the budget (idealized -- no "
              f"SCALE-Sim analog exists for these) ---")
        print(f"Idealized spill DRAM bytes (no SCALE-Sim analog): "
              f"{total_idealized_spill_bytes}")
        print(f"Idealized retrieve DRAM bytes (consumed by a non-conv "
              f"layer, no SCALE-Sim analog): {total_idealized_retrieve_bytes}")
        print(f"Real (SCALE-Sim-simulated) retrieve DRAM bytes: "
              f"{total_real_retrieve_bytes}")
        print(f"Baseline DRAM bytes (no unified SPM): {baseline_dram_bytes}")
        print(f"COSMA DRAM bytes: {cosma_dram_bytes}")
        print(f"DRAM traffic reduction: {dram_traffic_reduction_pct:.1f}%")
        print(f"Baseline total cycles: {baseline_total_cycles:.1f}")
        print(f"COSMA total cycles: {cosma_total_cycles:.1f}")
        print(f"Speedup: {speedup:.4f}x")
        print(f"--- Compute-vs-memory bound breakdown, per layer "
              f"(max(compute_cycles, dram_bytes/{bandwidth_bytes_per_cycle:.0f})) ---")
        print(f"Baseline: {baseline_memory_bound_layers}/{len(layer_bound_breakdown)} "
              f"layers memory-bound")
        print(f"COSMA:    {cosma_memory_bound_layers}/{len(layer_bound_breakdown)} "
              f"layers memory-bound")
        interesting = [r for r in layer_bound_breakdown
                       if r['baseline_bound'] == 'memory' or r['cosma_bound'] == 'memory']
        if interesting:
            print("Layers where memory was (or became) the bottleneck -- these are the "
                  "only ones where COSMA's DRAM reduction can show up as real speedup:")
            for r in interesting:
                print(f"  t={r['t']:>3} {r['op']:<16} "
                      f"baseline: compute={r['baseline_compute_cycles']:.0f} "
                      f"mem={r['baseline_mem_cycles']:.0f} [{r['baseline_bound']}]   "
                      f"cosma: compute={r['cosma_compute_cycles']:.0f} "
                      f"mem={r['cosma_mem_cycles']:.0f} [{r['cosma_bound']}]")
        else:
            print("No layer was ever memory-bound at this array/bandwidth config -- "
                  "COSMA's DRAM reduction has no bottleneck left to relieve here, "
                  "regardless of how large it is (see ITERATION_HISTORY.md's array-size "
                  "discussion).")
        if plot_path:
            print(f"Saved SPM occupancy comparison to {plot_path}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-json', default=DEFAULT_MODEL_JSON,
                         help='Path to model.json, or a raw .tflite file -- a .tflite '
                              'is auto-exported to model.json and cached under '
                              '--export-dir (same behavior as run_experiments.py).')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--budget-kb', type=float, default=128)
    parser.add_argument('--time-limit', type=float, default=360)
    parser.add_argument('--no-plot', action='store_true',
                         help="Skip saving the baseline-vs-COSMA occupancy PNG "
                              "(visualize_spm.py's diagram, generated by default).")
    parser.add_argument('--plot-out', default=None,
                         help='Override the plot output path (default: '
                              'cosma/spm_plots/<model>_<budget>kb.png).')
    parser.add_argument('--exporter', default=model_resolver.DEFAULT_EXPORTER,
                         help='Path to trim/python_scripts/export_model.py '
                              '(only needed for .tflite inputs).')
    parser.add_argument('--export-dir', default=model_resolver.DEFAULT_EXPORT_DIR,
                         help='Cache directory for .tflite -> model.json exports.')
    parser.add_argument('--force-export', action='store_true',
                         help='Re-export even if a cached model.json exists.')
    args = parser.parse_args()

    run_cosma(
        model_json_path=args.model_json,
        config_path=args.config,
        memory_budget_bytes=int(args.budget_kb * 1024),
        ilp_time_limit_sec=args.time_limit,
        save_plot=not args.no_plot,
        plot_out_path=args.plot_out,
        exporter=args.exporter,
        export_dir=args.export_dir,
        force_export=args.force_export,
    )
