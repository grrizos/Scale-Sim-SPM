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

import graph_builder
import baseline
import cosma_Ilp

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


def run_cosma(model_json_path: str = DEFAULT_MODEL_JSON,
              config_path: str = DEFAULT_CONFIG,
              memory_budget_bytes: int = 128 * 1024,
              bandwidth_bytes_per_cycle: float = None,
              ilp_time_limit_sec: float = 120,
              layer_stats: dict = None,
              verbose: bool = True) -> dict:
    """
    layer_stats: optional precomputed baseline.run_baseline() output, to
        skip re-running SCALE-Sim (which doesn't depend on memory_budget_bytes
        at all, so callers sweeping several budgets for the same model
        should compute it once and pass it in -- see run_experiments.py).
    """
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

        baseline_total_cycles += max(base_s['compute_cycles'],
                                      baseline_compulsory / bandwidth_bytes_per_cycle)
        cosma_total_cycles += max(cosma_s['compute_cycles'],
                                   cosma_compulsory / bandwidth_bytes_per_cycle)
        baseline_dram_bytes += baseline_compulsory
        cosma_dram_bytes += cosma_compulsory

    dram_traffic_reduction_pct = (
        100.0 * (baseline_dram_bytes - cosma_dram_bytes) / baseline_dram_bytes
        if baseline_dram_bytes > 0 else 0.0
    )
    speedup = (baseline_total_cycles / cosma_total_cycles
               if cosma_total_cycles > 0 else float('inf'))

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

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-json', default=DEFAULT_MODEL_JSON)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--budget-kb', type=float, default=128)
    parser.add_argument('--time-limit', type=float, default=120)
    args = parser.parse_args()

    run_cosma(
        model_json_path=args.model_json,
        config_path=args.config,
        memory_budget_bytes=int(args.budget_kb * 1024),
        ilp_time_limit_sec=args.time_limit,
    )
