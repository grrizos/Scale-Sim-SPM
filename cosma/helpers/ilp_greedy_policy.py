# cosma/helpers/ilp_greedy_policy.py
"""
The paper's "ILP-based greedy algorithm" tensor-replacement baseline
(§V-A2): "an ILP-based greedy algorithm which provides a locally optimal
decision that generates the least off-chip data accesses each time
replacement is needed." A genuinely LOCAL ILP -- one small solve per
eviction decision point, not a global solve over the whole schedule like
COSMA's own joint ILP (cosma_Ilp.build_cosma_model()).

Implements the choose_victims_fn contract shared with belady_policy.py --
see replacement_engine.simulate_replacement()'s docstring. Called only
with `candidates` already restricted (by the engine's own dead-tensor
pass) to tensors with a real, guaranteed future consumer.

Decision variables: x[a] in {0,1} per candidate a (1 = evict a now).
Objective: minimize sum(x[a] * size(a)).
Constraint: sum(x[a] * size(a)) >= deficit_bytes (free at least enough).

Why "minimize total evicted bytes" is a faithful (not approximate)
translation of "least off-chip data accesses," not a simplification of it:
every candidate is guaranteed (by the engine's dead-tensor pass) to have a
real future consumer at some position p > current_t. Evicting it now
deterministically costs one Spill now (size(a) bytes) AND triggers exactly
one future Retrieve (size(a) bytes) when the schedule reaches p and finds
it non-resident -- barring a further eviction before p, which can only ADD
more S+R pairs, never remove this one. So the true eventual off-chip cost
of evicting a now is 2*size(a) at minimum, and since that multiplier is
the SAME constant for every candidate, ranking/choosing by size(a) alone
is order-equivalent to ranking by true eventual cost -- while staying
genuinely LOCAL (no lookahead into whether an evicted tensor might itself
get evicted again before use -- that's the GLOBAL reasoning COSMA's own
joint ILP is for, and greedy is deliberately not).

Why this predicts the paper's own reported finding (Belady worse than
greedy): Belady ranks purely by TIMING (size-blind); this policy ranks
purely by SIZE (timing-blind). Concrete divergence, verified in this
module's own __main__: deficit=100KB, candidates X(150KB, far-future),
Y(60KB, near-future), Z(60KB, near-future). Belady evicts {X} alone
(furthest future) -> 150KB evicted -> 300KB eventual off-chip traffic.
This policy evicts {Y,Z} (120KB, minimal covering total, 120 < 150) ->
240KB eventual off-chip traffic -- strictly less, despite evicting the
SOONER-needed tensors.
"""
from typing import Dict, List

import pulp

from .replacement_engine import ReplacementInfeasible


def _solve(prob, time_limit_sec, solver, msg=False):
    """
    Deliberately NOT imported from cosma_Ilp.solve() -- same CBC/Gurobi
    dispatch pattern, but kept local so this module has zero coupling to
    cosma_Ilp.py (see docs/baseline_construction.md's isolation
    rationale). Raises ValueError for an unrecognized solver, same as
    cosma_Ilp.solve().

    Returns (status, has_feasible_incumbent) -- same meaning as
    cosma_Ilp.solve()'s return value: has_feasible_incumbent is True only
    for solver='gurobi' when the solve stopped short of proving optimality
    (time limit, or interrupted via Ctrl+C -- gurobipy's optimize() traps
    SIGINT itself and returns normally with GRB.INTERRUPTED rather than
    raising) but Gurobi still has an integer-feasible solution in hand
    (model.SolCount >= 1). See cosma_Ilp.solve()'s docstring for the full
    rationale, including why PuLP has already populated every variable's
    .varValue with that incumbent by the time this returns.
    """
    if solver == 'gurobi':
        pulp_solver = pulp.GUROBI(msg=msg, timeLimit=time_limit_sec)
    elif solver == 'cbc':
        pulp_solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec)
    else:
        raise ValueError(f"Unknown solver {solver!r} -- expected 'cbc' or 'gurobi'")
    prob.solve(pulp_solver)
    status = pulp.LpStatus[prob.status]
    has_feasible_incumbent = (
        solver == 'gurobi' and status != 'Optimal'
        and getattr(prob.solverModel, 'SolCount', 0) >= 1
    )
    return status, has_feasible_incumbent


def choose_victims(candidates: List[int], deficit_bytes: int, current_t: int,
                    tensors: Dict[int, object],
                    consumer_positions: Dict[int, List[int]],
                    time_limit_sec: float = 10, solver: str = 'cbc') -> List[int]:
    """
    consumer_positions is accepted (for choose_victims_fn contract parity
    with belady_policy.choose_victims) but not used -- this policy's
    objective only needs candidate sizes, not their timing.

    Fast pre-check before paying for a solve, mirroring
    cosma_Ilp.assert_tensors_fit_budget()'s own "cheap failure before an
    expensive step" pattern: if even evicting every candidate can't cover
    the deficit, return [] immediately -- replacement_engine.py turns an
    empty return into ReplacementInfeasible, so this never needs to raise
    that itself for this specific case.

    time_limit_sec defaults to a small 10s safety net (unlike
    cosma_Ilp.solve()'s unbounded default) since this may be invoked many
    times per run, not once per whole-graph solve -- a single pathological
    candidate set hanging here shouldn't be able to hang an entire sweep
    the way an unbounded whole-graph COSMA solve legitimately can.
    """
    if not candidates:
        return []
    if sum(tensors[a].size_bytes for a in candidates) < deficit_bytes:
        return []

    prob = pulp.LpProblem("cosma_baseline_ilp_greedy_replacement", pulp.LpMinimize)
    x = {a: pulp.LpVariable(f"evict_{a}_{current_t}", cat='Binary') for a in candidates}
    cost = pulp.lpSum(x[a] * tensors[a].size_bytes for a in candidates)
    prob += cost
    prob += cost >= deficit_bytes, "cover_deficit"

    status, has_feasible_incumbent = _solve(
        prob, time_limit_sec=time_limit_sec, solver=solver)
    if status != 'Optimal' and not has_feasible_incumbent:
        # The pre-check above already guarantees this sub-problem is
        # feasible (set every x[a]=1 -> cost = sum(sizes) >= deficit,
        # always satisfies the one constraint) -- so a non-Optimal status
        # here can only mean the solver didn't finish in time, never a
        # real infeasibility. Same CBC/PuLP mislabeling risk already
        # documented (and hit for real, on _exported/fake2/) in
        # cosma_Ilp.compute_true_mpmf_bytes()'s docstring applies here
        # identically -- don't trust an 'Infeasible' status from CBC under
        # a time limit as a proof; it can mean nothing more than "ran out
        # of time."
        caveat = (
            " -- this sub-problem is always feasible by construction (the "
            "pre-check above already confirmed candidates can cover the "
            "deficit), so this status can only mean the solver didn't finish "
            "in time -- CBC/PuLP can mislabel that 'Infeasible' rather than "
            "'Not Solved'. Retry with a longer time_limit_sec, or use "
            "solver='gurobi' for reliable status reporting."
            if solver == 'cbc' else ""
        )
        raise ReplacementInfeasible(
            f"ilp_greedy_policy: local eviction sub-problem at t={current_t} did "
            f"not solve to optimality (status={status}, {len(candidates)} "
            f"candidates, deficit={deficit_bytes} bytes){caveat}",
            timestep=current_t, reason='local_ilp_not_optimal')
    if status != 'Optimal':
        print(f"WARNING: ilp_greedy_policy: local eviction sub-problem at "
              f"t={current_t} did not prove optimality (status={status}) -- "
              f"accepting Gurobi's best incumbent found so far instead.")

    return [a for a in candidates if round(pulp.value(x[a])) == 1]


if __name__ == '__main__':
    from . import belady_policy

    class _T:
        def __init__(self, size_bytes):
            self.size_bytes = size_bytes

    # The exact divergence scenario from this module's own docstring:
    # deficit=100KB, X=150KB far-future, Y=Z=60KB near-future.
    tensors = {
        'X': _T(150 * 1024),
        'Y': _T(60 * 1024),
        'Z': _T(60 * 1024),
    }
    consumer_positions = {'X': [100], 'Y': [5], 'Z': [6]}
    candidates = ['X', 'Y', 'Z']
    deficit = 100 * 1024
    current_t = 0

    belady_victims = belady_policy.choose_victims(
        candidates, deficit, current_t, tensors, consumer_positions)
    greedy_victims = choose_victims(
        candidates, deficit, current_t, tensors, consumer_positions)

    belady_bytes = sum(tensors[a].size_bytes for a in belady_victims)
    greedy_bytes = sum(tensors[a].size_bytes for a in greedy_victims)

    print(f"Belady evicts {sorted(belady_victims)}: {belady_bytes} bytes")
    print(f"ILP-greedy evicts {sorted(greedy_victims)}: {greedy_bytes} bytes")

    assert sorted(belady_victims) == ['X'], f"expected Belady to evict just X, got {belady_victims}"
    assert sorted(greedy_victims) == ['Y', 'Z'], f"expected greedy to evict Y and Z, got {greedy_victims}"
    assert greedy_bytes < belady_bytes, "expected greedy's eviction to cost fewer bytes than Belady's"
    print(f"PASS: greedy evicts strictly fewer bytes ({greedy_bytes} < {belady_bytes}), "
          f"reproducing the paper's own reported Belady-suboptimal finding.")

    # Cross-check on toy_spill_model.json (via replacement_engine.py): only
    # one candidate is EVER evictable at a time in that fixture, so both
    # policies must agree exactly -- an independent regression check
    # against replacement_engine.py's own __main__ hand-derivation.
    import os
    from . import graph_builder, replacement_engine

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nodes, tensors2 = graph_builder.load_graph(os.path.join(here, 'toy_spill_model.json'))
    schedule = [(0, 0), (1, 1), (2, 2), (3, 3)]

    belady_result = replacement_engine.simulate_replacement(
        nodes, tensors2, schedule, 200, belady_policy.choose_victims)
    greedy_result = replacement_engine.simulate_replacement(
        nodes, tensors2, schedule, 200, choose_victims)
    assert belady_result == greedy_result, (
        f"toy_spill_model.json has only one evictable candidate at a time -- "
        f"both policies must agree exactly, but got:\n"
        f"  belady: {belady_result}\n  greedy: {greedy_result}")
    print("toy_spill_model.json cross-check: PASS, Belady and ILP-greedy agree "
          "exactly (only one candidate ever evictable in this fixture)")
