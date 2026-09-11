# cosma/helpers/schedule_variants.py
"""
Produces the two fixed operator schedules the paper's comparison
baselines need (§V-A2, "Operator Scheduling"): "(i) The default operator
scheduling from the PyTorch implementation of the DNNs, and (ii) the
operator schedule with minimum peak memory footprint (MPMF) of the DNN."

Both returned as [(t, layer_id), ...] sorted by t -- the exact shape
helpers.baseline.run_cosma_aware()'s `schedule` param and
helpers.replacement_engine.simulate_replacement()'s `schedule` param both
need (same shape cosma_Ilp.extract_results()['schedule_layer_at_t'].items()
already produces for the main pipeline's free_schedule=True mode).

Only file (besides the driver script) that imports from cosma_Ilp.py, and
only its already-public, already-multi-caller compute_true_mpmf_bytes() --
read-only, no mutation, no new coupling surface (visualize_spm.py already
calls this same function for --true-mpmf).
"""
from typing import Dict, List, Tuple

from . import cosma_Ilp


def default_operator_schedule(nodes: Dict[int, object]) -> List[Tuple[int, int]]:
    """
    The paper's "default operator scheduling from the PyTorch
    implementation" -- model.json's own topological order, t == position
    == layer id. Built explicitly (not relying on
    helpers.baseline._run_layers()'s implicit None-default) so both
    schedule variants this module produces are constructed the same,
    visible way.
    """
    return [(t, layer_id) for t, layer_id in enumerate(sorted(nodes.keys()))]


def mpmf_operator_schedule(nodes: Dict[int, object], tensors: Dict[int, object],
                            mpmf_tensor_schedule: Dict[int, int]) -> List[Tuple[int, int]]:
    """
    The paper's "operator schedule with minimum peak memory footprint
    (MPMF)" -- explicitly the §III-E1/Eq.13-15 schedule per the paper's
    own text ("MPMF is calculated: (i) by using COSMA for the
    human-designed... DNNs (§III-E1)"), i.e. exactly what
    cosma_Ilp.compute_true_mpmf_bytes() already computes. NOT always the
    same as default_operator_schedule() -- confirmed empirically to
    genuinely reorder on toy_spill_model.json (see this module's own
    __main__), so don't assume equality without checking.

    mpmf_tensor_schedule: the `schedule` dict compute_true_mpmf_bytes()
        returns -- {tensor_id: t}, keyed by TENSOR id (which timestep that
        tensor's producing node was assigned). This function inverts it to
        the node/layer-keyed [(t, layer_id), ...] shape every other
        consumer in this codebase expects, via each tensor's own
        producer_layer (a structural fact, independent of scheduling).

    Asserts the result is a full bijection over sorted(nodes.keys()) --
    cheap, standing-practice insurance (see this project's "verify
    empirically, don't just trust the encoding" practice): guaranteed by
    build_mpmf_schedule_model()'s own Eq.7 + "at most one node per
    timestep" constraint for any Optimal solve, but checked here rather
    than assumed, so a future change to that ILP's constraints would be
    caught here instead of silently corrupting a schedule.
    """
    schedule_layer_at_t: Dict[int, int] = {}
    for tensor_id, t in mpmf_tensor_schedule.items():
        layer_id = tensors[tensor_id].producer_layer
        # Siblings (tensors sharing a producer node) collide onto the same
        # t harmlessly -- Eq.6 already guarantees they agree, so either
        # assigning first-wins or overwriting lands on the same layer_id.
        schedule_layer_at_t[t] = layer_id

    expected_nodes = set(nodes.keys())
    expected_ts = set(range(len(nodes)))
    got_ts = set(schedule_layer_at_t.keys())
    got_nodes = set(schedule_layer_at_t.values())
    assert got_ts == expected_ts, (
        f"mpmf_operator_schedule(): expected timesteps {sorted(expected_ts)}, "
        f"got {sorted(got_ts)} -- not a full bijection, a real bug in the "
        f"MPMF schedule ILP or this inversion")
    assert got_nodes == expected_nodes, (
        f"mpmf_operator_schedule(): expected every node {sorted(expected_nodes)} "
        f"to appear exactly once, got {sorted(got_nodes)} -- not a full "
        f"bijection, a real bug in the MPMF schedule ILP or this inversion")

    return sorted(schedule_layer_at_t.items())


if __name__ == '__main__':
    import os
    from . import graph_builder

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cosma/

    # toy_spill_model.json: hand-derived AND empirically confirmed (during
    # this module's own development -- see docs/baseline_construction.md's
    # verification log) that the true MPMF-optimal schedule genuinely
    # reorders relative to default: default creates tensor 10 (needed only
    # at the very end, by node 3) at t=0 and holds it resident the whole
    # time (peak 210B, matching the fixture's own documented
    # compute_mpmf_bytes proxy), whereas the MPMF-optimal order instead
    # runs node1->node2->node0->node3 (layers [1,2,0,3]), delaying tensor
    # 10's creation so it's never resident alongside both 11 and 12 at
    # once -- reaching exactly M_R (200B), confirmed by results_plan.md's
    # own model-roster table. This is a real, useful regression check
    # precisely because the two schedules differ.
    nodes, tensors = graph_builder.load_graph(os.path.join(here, 'toy_spill_model.json'))
    default_sched = default_operator_schedule(nodes)
    assert default_sched == [(0, 0), (1, 1), (2, 2), (3, 3)]

    mpmf_bytes, mpmf_tensor_sched = cosma_Ilp.compute_true_mpmf_bytes(nodes, tensors)
    mpmf_sched = mpmf_operator_schedule(nodes, tensors, mpmf_tensor_sched)
    print(f"toy_spill_model.json: default={default_sched}")
    print(f"toy_spill_model.json: MPMF={mpmf_sched} ({mpmf_bytes} bytes)")
    assert mpmf_bytes == 200, f"expected true M_P == M_R == 200, got {mpmf_bytes}"
    assert mpmf_sched == [(0, 1), (1, 2), (2, 0), (3, 3)], (
        f"expected the hand-derived/empirically-confirmed reorder "
        f"[(0,1),(1,2),(2,0),(3,3)], got {mpmf_sched}")
    assert mpmf_sched != default_sched
    print("toy_spill_model.json: MPMF genuinely reorders vs default, matches "
          "hand-derivation exactly, bijection verified. PASS")

    # Small custom DenseNet fixture: confirmed empirically (this session,
    # live run) that true_M_P == MPMF-proxy (622592 bytes both) for this
    # model -- so the MPMF schedule found here should be byte-identical to
    # default, with zero new computation needed to know the expected
    # answer. This is the opposite regression shape from toy_spill_model
    # above (equal, not reordered) -- both directions matter to check.
    fake_path = os.path.join(here, '_exported', 'fake', 'model.json')
    if os.path.exists(fake_path):
        nodes2, tensors2 = graph_builder.load_graph(fake_path)
        default_sched2 = default_operator_schedule(nodes2)
        mpmf_bytes2, mpmf_tensor_sched2 = cosma_Ilp.compute_true_mpmf_bytes(nodes2, tensors2)
        mpmf_sched2 = mpmf_operator_schedule(nodes2, tensors2, mpmf_tensor_sched2)
        assert mpmf_sched2 == default_sched2, (
            f"_exported/fake/model.json: expected MPMF schedule to match "
            f"default (confirmed empirically true_M_P == MPMF-proxy == "
            f"622592 for this model), got a real difference")
        print(f"_exported/fake/model.json ({mpmf_bytes2} bytes): MPMF schedule "
              f"== default schedule, as expected. PASS")
    else:
        print("(skipping _exported/fake/model.json check -- not exported here)")
