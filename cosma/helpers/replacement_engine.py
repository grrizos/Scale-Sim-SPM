# cosma/helpers/replacement_engine.py
"""
The shared WHAT/WHEN simulation loop for the paper's two tensor-replacement
baselines (Belady, ILP-greedy -- see belady_policy.py/ilp_greedy_policy.py).
Written once, here, so the two policies can only ever differ in WHICH
tensor(s) they pick to evict, never in the surrounding loop mechanics
(freeing dead tensors, admitting a node's demand as one combined batch,
when eviction is even triggered) -- avoiding exactly the kind of subtle
ordering-correctness drift that would be easy to introduce by duplicating
this loop in two files.

Produces resident_action: {(tensor_id, t): 'C'|'P'|'S'|'R'}, the same
vocabulary/shape cosma_Ilp.extract_results()['resident_action'] already
has -- a drop-in for helpers.tflite_arena_allocator.place_tensors_linear()
and helpers.baseline.run_cosma_aware().

Correctness note (found and fixed during development, not assumed correct
from the start -- see docs/baseline_construction.md's verification log):
tensors[a].consumer_layers stores LAYER ids, but this engine walks the
schedule by abstract TIMESTEP t -- under a reordered schedule (e.g.
schedule_variants.mpmf_operator_schedule()'s output), layer id and
timestep position are NOT the same number. "Has tensor a's last consumer
already run by timestep t" therefore cannot compare consumer_layers
against t directly -- it must go through the schedule's own layer_id -> t
mapping first. This engine precomputes that once (`position_of_layer`) and
derives `consumer_positions` (tensor_id -> sorted list of t-positions, NOT
layer ids) from it, passed to choose_victims_fn so Belady/ILP-greedy don't
have to re-derive it (or risk the same mistake) themselves.
"""
from typing import Dict, List, Tuple


class ReplacementInfeasible(RuntimeError):
    """
    Raised when a replacement policy cannot free enough space to admit a
    timestep's demand within memory_budget_bytes -- either the budget is
    below the model's own structural minimum (M_R), or (for ILP-greedy) the
    local sub-problem itself failed to solve. Not necessarily a bug -- see
    cosma_Ilp.compute_structural_minimum_bytes() for the real floor.
    """
    def __init__(self, message: str, *, timestep: int = None, reason: str = None):
        super().__init__(message)
        self.timestep = timestep
        self.reason = reason


def simulate_replacement(nodes: Dict[int, object], tensors: Dict[int, object],
                          schedule: List[Tuple[int, int]],
                          memory_budget_bytes: int,
                          choose_victims_fn) -> Dict[Tuple[int, int], str]:
    """
    Walks `schedule` ([(t, layer_id), ...], sorted by t) in order,
    producing resident_action for every C/P/S/R event.

    choose_victims_fn(candidates, deficit_bytes, current_t, tensors,
                       consumer_positions) -> List[tensor_id]: called only
    when genuinely short on space (never speculatively), with `candidates`
    already restricted to "currently resident, not needed by the op about
    to run" -- every candidate is guaranteed a real future consumer (dead
    tensors are freed for free, before this is ever called -- see below).
    Both belady_policy.choose_victims and ilp_greedy_policy.choose_victims
    implement this exact contract, so they're interchangeable here.
    """
    position_of_layer: Dict[int, int] = {lid: t for t, lid in schedule}

    consumer_positions: Dict[int, List[int]] = {}
    for a, tensor in tensors.items():
        positions = sorted(position_of_layer[c] for c in tensor.consumer_layers
                            if c in position_of_layer)
        consumer_positions[a] = positions

    occupied_bytes = 0
    resident_now = set()
    resident_action: Dict[Tuple[int, int], str] = {}
    spill_count = 0
    total_spilled_bytes = 0

    for (t, lid) in schedule:
        node = nodes[lid]

        # Step 0 -- free anything genuinely dead: no remaining consumer at
        # or after this position. Free, deterministic, no policy involved
        # -- evicting something with no remaining use is strictly dominant
        # over evicting something still needed, always (matches the
        # paper's own model: "dead tensors are deallocated immediately
        # after execution ... modeled by setting their preservation
        # variable to zero" -- no S recorded, an implicit lapse, exactly
        # spm_allocator.py's own documented "implicit lapse, not just
        # explicit Spill" behavior).
        for a in list(resident_now):
            remaining = [p for p in consumer_positions[a] if p >= t]
            if not remaining:
                resident_now.discard(a)
                occupied_bytes -= tensors[a].size_bytes

        # Step 1 -- this timestep's demand batch: activation inputs not
        # already resident need a Retrieve; the node's own outputs are
        # always a Create. `protected` covers both regardless of whether
        # each is newly admitted or already resident -- neither can be
        # evicted to make room for the other (they're needed by the same
        # op, right now).
        protected = ({a for a in node.activation_inputs if a in tensors}
                     | {a for a in node.outputs if a in tensors})
        admit_batch = (
            [(a, 'R') for a in node.activation_inputs
             if a in tensors and a not in resident_now]
            + [(a, 'C') for a in node.outputs if a in tensors]
        )
        batch_bytes = sum(tensors[a].size_bytes for a, _ in admit_batch)

        # Step 2 -- evict only if short on space, as one combined batch
        # (not per-input) so a node's own inputs/outputs never compete
        # with each other as eviction targets.
        deficit = occupied_bytes + batch_bytes - memory_budget_bytes
        while deficit > 0:
            candidates = [x for x in resident_now if x not in protected]
            victims = choose_victims_fn(candidates, deficit, t, tensors, consumer_positions)
            freed = sum(tensors[v].size_bytes for v in victims)
            if not victims or freed == 0:
                raise ReplacementInfeasible(
                    f"cannot free {deficit} more bytes at t={t} (layer {lid}) "
                    f"within the {memory_budget_bytes}-byte budget -- infeasible "
                    f"for this schedule/policy (or budget below the model's own "
                    f"structural minimum, cosma_Ilp.compute_structural_minimum_bytes())",
                    timestep=t, reason='cannot_free_enough')
            for v in victims:
                resident_now.discard(v)
                occupied_bytes -= tensors[v].size_bytes
                resident_action[(v, t)] = 'S'
                spill_count += 1
                total_spilled_bytes += tensors[v].size_bytes
            deficit = occupied_bytes + batch_bytes - memory_budget_bytes

        for a, action in admit_batch:
            resident_now.add(a)
            occupied_bytes += tensors[a].size_bytes
            resident_action[(a, t)] = action

        # Step 3 -- everyone else still resident gets Preserve. setdefault
        # so a tensor this same t's admit_batch already gave 'C'/'R' to
        # (added to resident_now just above) is never overwritten.
        for a in resident_now:
            resident_action.setdefault((a, t), 'P')

    print(f"[replacement_engine] {len(schedule)} timesteps, {spill_count} spills, "
          f"{total_spilled_bytes} bytes spilled, 0 infeasibilities")

    return resident_action


if __name__ == '__main__':
    import os

    def _fixed_deficit_victims(candidates, deficit_bytes, current_t, tensors, consumer_positions):
        """Trivial policy for this smoke test: evict in id order until covered
        -- toy_spill_model.json's default schedule only ever has exactly one
        evictable candidate at a time, so any policy must agree (see
        belady_policy.py/ilp_greedy_policy.py's own smoke tests for the
        real policies, and their cross-check against this exact fixture)."""
        victims, freed = [], 0
        for a in sorted(candidates):
            if freed >= deficit_bytes:
                break
            victims.append(a)
            freed += tensors[a].size_bytes
        return victims

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from . import graph_builder
    nodes, tensors = graph_builder.load_graph(os.path.join(here, 'toy_spill_model.json'))
    schedule = [(0, 0), (1, 1), (2, 2), (3, 3)]  # default schedule

    result = simulate_replacement(nodes, tensors, schedule, 200, _fixed_deficit_victims)
    expected = {
        (10, 0): 'C', (10, 1): 'P', (10, 2): 'S', (10, 3): 'R',
        (11, 1): 'C', (11, 2): 'P',
        (12, 2): 'C', (12, 3): 'P',
        (13, 3): 'C',
    }
    print("resident_action:", dict(sorted(result.items(), key=lambda kv: (kv[0][1], kv[0][0]))))
    assert result == expected, f"mismatch:\n  got:      {result}\n  expected: {expected}"
    print("toy_spill_model.json, budget=200: PASS, matches hand-derived resident_action exactly")

    # Below M_R (200): must raise ReplacementInfeasible, not hang or return
    # a wrong/silent answer.
    try:
        simulate_replacement(nodes, tensors, schedule, 110, _fixed_deficit_victims)
        raise AssertionError("expected ReplacementInfeasible for budget=110 (< M_R=200), got none")
    except ReplacementInfeasible as e:
        print(f"budget=110 (< M_R=200): PASS, correctly raised: {e}")
