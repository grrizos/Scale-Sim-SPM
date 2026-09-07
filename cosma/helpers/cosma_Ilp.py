# cosma/helpers/cosma_Ilp.py
"""
Builds and solves the COSMA ILP (Eq. 1-12 of the plan doc) given a graph
from graph_builder.py.

Scope decision: the operator schedule (timestep order) is FIXED to
model.json's topological order (timestep t == layer id -- confirmed via
model.json's own `topo_sort.order`, which is the identity permutation for
this graph). SCALE-Sim is re-run per that same fixed layer order (see the
plan doc's risk table: "SCALE-Sim layer order is fixed -> Cannot reorder").

Because the schedule is fixed, C[a,t] ("tensor a created at t") is a
*constant*, not a decision variable: a is always created at
t = producer_layer(a). This means:
  - Eq.6 (sibling tensors created together) is satisfied by construction --
    siblings share a producer node, hence the same fixed producer_layer.
  - Eq.7 (each tensor created exactly once) is satisfied by construction.
  - The N*T binary C variables are eliminated entirely, which materially
    helps the "ILP grows quadratically" risk called out in the plan doc.

What the ILP actually decides is Memory Allocation + Tensor Replacement:
per tensor per timestep, whether to preserve in SPM (P), spill to DRAM (S),
retrieve from DRAM (R), and (while resident) its base address (L) --
subject to the non-overlap (Eq.10) and budget (Eq.9) constraints, filtered
to only tensor pairs whose liveness windows actually overlap.

Also exposes two solve-free feasibility bounds (compute_structural_minimum_bytes,
compute_mpmf_bytes) -- the budget range worth searching at all, without
paying for a full build_cosma_model()/solve().
"""
import pulp
from typing import Dict, List, Tuple


def _liveness_windows(tensors) -> Dict[int, Tuple[int, int]]:
    """(first timestep a tensor could be resident, last timestep it's needed)."""
    windows = {}
    for tid, t in tensors.items():
        start = t.producer_layer
        end = max(t.consumer_layers) if t.consumer_layers else start
        windows[tid] = (start, end)
    return windows


def compute_structural_minimum_bytes(nodes, tensors) -> Tuple[int, int]:
    """
    The paper's M_R: the largest amount of tensor memory that MUST be
    simultaneously resident at some single timestep no matter what
    scheduling/spilling choices are made -- for each node t, the sum of
    the sizes of its own activation_inputs plus its own outputs (exactly
    what Eq.5 already forces resident at t = producer_t[a]). Below this
    budget, build_cosma_model()/solve() is provably Infeasible -- no need
    to build or solve the ILP to find that out.

    Returns (bytes, argmax_timestep) so a caller can report *which*
    operator is the bottleneck, not just the number.
    """
    floor_bytes, floor_t = 0, None
    for t in sorted(nodes.keys()):
        node = nodes[t]
        live = sum(tensors[a].size_bytes for a in node.activation_inputs if a in tensors)
        live += sum(tensors[a].size_bytes for a in node.outputs if a in tensors)
        if live > floor_bytes:
            floor_bytes, floor_t = live, t
    return floor_bytes, floor_t


def compute_mpmf_bytes(nodes, tensors) -> Tuple[int, int]:
    """
    The paper's MPMF (minimum peak memory footprint): the peak, over all
    timesteps t, of the combined size of every tensor whose liveness
    window (_liveness_windows) covers t. At or above this budget, Eq.12's
    objective is always 0 -- everything that could ever coexist already
    fits, so nothing is ever evicted.

    Returns (bytes, argmax_timestep).
    """
    windows = _liveness_windows(tensors)
    ceiling_bytes, ceiling_t = 0, None
    for t in sorted(nodes.keys()):
        live = sum(tensors[tid].size_bytes for tid, (start, end) in windows.items()
                   if start <= t <= end)
        if live > ceiling_bytes:
            ceiling_bytes, ceiling_t = live, t
    return ceiling_bytes, ceiling_t


def assert_tensors_fit_budget(tensors, memory_budget_bytes: int) -> None:
    """
    Fast, baseline-independent feasibility pre-check: no single tensor can
    exceed the budget on its own, regardless of scheduling or spilling.

    Cosma will fail if  a tensor is too big for the budget, but this pre-check 
    is cheaper and more user-friendly than waiting for the ILP to fail.
    """
    for a, t in tensors.items():
        assert t.size_bytes <= memory_budget_bytes, (
            f"tensor {a} ({t.size_bytes} bytes) does not fit in the "
            f"{memory_budget_bytes}-byte SPM budget on its own -- "
            f"infeasible regardless of scheduling"
        )


def build_cosma_model(nodes, tensors, memory_budget_bytes: int):
    """
    nodes:   dict[int, graph_builder.Node]
    tensors: dict[int, graph_builder.Tensor]  (COSMA-tracked activations only)
    memory_budget_bytes: SPM capacity, also used as the big-M constant
        (per the plan doc's Eq.10 risk mitigation: "Set M = memory_budget,
        not a huge arbitrary number").

    Returns (prob, variables, T, A) where variables is a dict of the
    PuLP variable dicts (P, S, R, L, u, d) keyed as described below.
    """
    T = sorted(nodes.keys())
    A = sorted(tensors.keys())
    budget = memory_budget_bytes
    size = {a: tensors[a].size_bytes for a in A}
    producer_t = {a: tensors[a].producer_layer for a in A}
    windows = _liveness_windows(tensors)

    assert_tensors_fit_budget(tensors, budget)

    prob = pulp.LpProblem("COSMA", pulp.LpMinimize)

    def C(a, t):
        return 1 if t == producer_t[a] else 0

    P = {(a, t): pulp.LpVariable(f"P_{a}_{t}", cat='Binary') for a in A for t in T}
    S = {(a, t): pulp.LpVariable(f"S_{a}_{t}", cat='Binary') for a in A for t in T}
    R = {(a, t): pulp.LpVariable(f"R_{a}_{t}", cat='Binary') for a in A for t in T}
    L = {(a, t): pulp.LpVariable(f"L_{a}_{t}", lowBound=0, upBound=budget, cat='Integer')
         for a in A for t in T}

    def resident(a, t):
        return C(a, t) + P[a, t] + R[a, t]

    # Eq.1: at most one action per tensor per timestep
    for a in A:
        for t in T:
            prob += C(a, t) + P[a, t] + S[a, t] + R[a, t] <= 1, f"Eq1_{a}_{t}"

    # Eq.2/3: preserve/spill only if resident at t-1; Eq.4: retrieve only if spilled by t.
    # Base case (t == T[0]): there is no t-1 for Eq.2/3 to chain from, so
    # nothing can be "already resident" yet -- without this, P[a,T[0]]/
    # S[a,T[0]] are left completely unconstrained for every tensor (Eq.1
    # alone doesn't forbid them), letting the solver plant a zero-cost
    # "phantom" P (or, worse, a genuinely double-counted S) on a tensor
    # before it's even created. Confirmed to actually happen on ResNet-50
    # (5 of 79 tensors got a spurious P at t=0 in an optimal solve) --
    # silent on every previously-validated smaller model, but a real gap.
    t0 = T[0]
    for a in A:
        for t in T:
            if t > t0:
                prob += P[a, t] <= C(a, t - 1) + P[a, t - 1] + R[a, t - 1], f"Eq2_{a}_{t}"
                prob += S[a, t] <= C(a, t - 1) + P[a, t - 1], f"Eq3_{a}_{t}"
            else:
                prob += P[a, t] == 0, f"Eq2_base_{a}_{t}"
                prob += S[a, t] == 0, f"Eq3_base_{a}_{t}"
            prob += R[a, t] <= pulp.lpSum(S[a, k] for k in T if k <= t), f"Eq4_{a}_{t}"

    # Eq.5: an operator's activation inputs must already be resident when it runs.
    # (Schedule is fixed, so this only needs checking at t = producer_t[a].)
    for a in A:
        t = producer_t[a]
        node = nodes[t]
        for b in node.activation_inputs:
            if b in tensors:
                prob += C(a, t) <= P[b, t] + R[b, t], f"Eq5_{a}_{b}_{t}"

    # Eq.6 and Eq.7 hold by construction (see module docstring).

    # Eq.8: each tensor spilled at most once
    for a in A:
        prob += pulp.lpSum(S[a, t] for t in T) <= 1, f"Eq8_{a}"

    # Eq.9: resident tensor must fit the budget (relaxed to a no-op when not resident)
    for a in A:
        for t in T:
            prob += (L[a, t] + size[a] <= budget + budget * (1 - resident(a, t)),
                     f"Eq9_{a}_{t}")

    # Overlap-filtered tensor pairs for Eq.10 (the quadratic-blowup mitigation).
    pairs: List[Tuple[int, int]] = []
    for i, a in enumerate(A):
        pa_start, pa_end = windows[a]
        for b in A[i + 1:]:
            pb_start, pb_end = windows[b]
            if pa_start <= pb_end and pb_start <= pa_end:
                pairs.append((a, b))

    u: Dict[Tuple[int, int, int], pulp.LpVariable] = {}
    d: Dict[Tuple[int, int, int], pulp.LpVariable] = {}
    for (a, b) in pairs:
        lo = min(windows[a][0], windows[b][0])
        hi = max(windows[a][1], windows[b][1])
        for t in T:
            if lo <= t <= hi:
                u[a, b, t] = pulp.LpVariable(f"u_{a}_{b}_{t}", cat='Binary')
                d[a, b, t] = pulp.LpVariable(f"d_{a}_{b}_{t}", cat='Binary')

    # Eq.10: no two simultaneously-resident tensors may overlap in SPM address space.
    for (a, b) in pairs:
        for t in T:
            if (a, b, t) not in u:
                continue
            prob += (L[a, t] >= L[b, t] + size[b] - budget * (1 - u[a, b, t]),
                     f"Eq10u_{a}_{b}_{t}")
            prob += (L[b, t] >= L[a, t] + size[a] - budget * (1 - d[a, b, t]),
                     f"Eq10d_{a}_{b}_{t}")
            prob += (u[a, b, t] + d[a, b, t] >= resident(a, t) + resident(b, t) - 1,
                     f"Eq10c_{a}_{b}_{t}")
            prob += (u[a, b, t] + d[a, b, t] <= 1, f"Eq10mutex_{a}_{b}_{t}")

    # Eq.11: a tensor's address is pinned for as long as it stays continuously
    # preserved (P). V[a,t] is therefore just an alias for P[a,t] in this
    # fixed-schedule model, not an independent decision.
    V = P
    for a in A:
        for t in T:
            if t > 0:
                prob += (L[a, t] - L[a, t - 1] <= budget * (1 - V[a, t]),
                         f"Eq11a_{a}_{t}")
                prob += (L[a, t - 1] - L[a, t] <= budget * (1 - V[a, t]),
                         f"Eq11b_{a}_{t}")

    # Eq.12: minimize non-compulsory (spill + retrieve) DRAM traffic.
    prob += pulp.lpSum((S[a, t] + R[a, t]) * size[a] for a in A for t in T)

    variables = {'P': P, 'S': S, 'R': R, 'L': L, 'u': u, 'd': d, 'C': C}
    return prob, variables, T, A


def solve(prob, time_limit_sec=None, msg=False):
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec)
    prob.solve(solver)
    return pulp.LpStatus[prob.status]


def extract_results(variables, T, A, tensors):
    """
    Returns:
      ordered_layers:    fixed layer execution order (schedule is not
                          re-optimized in this implementation -- see
                          build_cosma_model's docstring)
      extra_dram_bytes:  timestep -> non-compulsory (spill+retrieve) bytes,
                          using COSMA's own idealized size(a) for every
                          event. Kept for backward compatibility / the
                          simple case; run_cosma.py should prefer
                          resident_action for a real-traffic-aware
                          breakdown (see its module docstring).
      spm_plan:          (tensor_id, timestep) -> base address, for every
                          timestep the tensor is resident
      resident_action:   (tensor_id, timestep) -> 'C'|'P'|'R'|'S', the
                          single action taken for that tensor at that
                          timestep (only present when one of the four is
                          1 -- Eq.1 guarantees at most one is). This is
                          what lets a caller distinguish "free" (P, or C
                          via Eq.3's reasoning that creation never needs a
                          DRAM round-trip), "a real retrieve happened" (R,
                          substitutable with a real simulated cost), and
                          "spilled" (S, no real-traffic equivalent exists
                          -- see run_cosma.py).
    """
    P, S, R, L, C = (variables['P'], variables['S'], variables['R'],
                     variables['L'], variables['C'])

    def val(x):
        v = pulp.value(x)
        return 0 if v is None else int(round(v))

    ordered_layers = list(T)

    extra_dram_bytes = {t: 0 for t in T}
    resident_action = {}
    for a in A:
        size = tensors[a].size_bytes
        for t in T:
            if C(a, t):
                resident_action[(a, t)] = 'C'
            elif val(P[a, t]):
                resident_action[(a, t)] = 'P'
            elif val(R[a, t]):
                resident_action[(a, t)] = 'R'
            elif val(S[a, t]):
                resident_action[(a, t)] = 'S'

            if val(S[a, t]) or val(R[a, t]):
                extra_dram_bytes[t] += size

    spm_plan = {}
    for a in A:
        for t in T:
            if C(a, t) or val(P[a, t]) or val(R[a, t]):
                spm_plan[(a, t)] = val(L[a, t])

    return {
        'ordered_layers': ordered_layers,
        'extra_dram_bytes': extra_dram_bytes,
        'spm_plan': spm_plan,
        'resident_action': resident_action,
    }
