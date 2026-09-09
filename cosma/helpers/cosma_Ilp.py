# cosma/helpers/cosma_Ilp.py
"""
Builds and solves the COSMA ILP (Eq. 1-12 of the plan doc) given a graph
from graph_builder.py.

build_cosma_model()'s default (free_schedule=False) keeps the operator
schedule (timestep order) FIXED to model.json's topological order
(timestep t == layer id -- confirmed via model.json's own `topo_sort.order`,
which is the identity permutation for this graph). Because the schedule is
fixed, C[a,t] ("tensor a created at t") is a *constant*, not a decision
variable: a is always created at t = producer_layer(a). This means:
  - Eq.6 (sibling tensors created together) is satisfied by construction --
    siblings share a producer node, hence the same fixed producer_layer.
  - Eq.7 (each tensor created exactly once) is satisfied by construction.
  - The N*T binary C variables are eliminated entirely, which materially
    helps the "ILP grows quadratically" risk called out in the plan doc.

Passing free_schedule=True turns C[a,t] into a real decision instead --
this is the paper's other headline half, "Combined **Scheduling**...",
previously implemented only inside the isolated build_mpmf_schedule_model()
(which never touches SPM placement or gets fed back into SCALE-Sim). With
free_schedule=True, the same C-is-free machinery is combined with the full
Eq.9-11 placement/replacement machinery below, and the resulting schedule
(extract_results()'s schedule_layer_at_t) is meant to drive a real
re-simulation (see baseline.py's run_cosma_aware() `schedule` param and
run_cosma.py). See build_cosma_model()'s own docstring for the constraint-
by-constraint detail and expected solve-time cost.

What the ILP decides (always): Memory Allocation + Tensor Replacement --
per tensor per timestep, whether to preserve in SPM (P), spill to DRAM (S),
retrieve from DRAM (R), and (while resident) its base address (L) --
subject to the non-overlap (Eq.10) and budget (Eq.9) constraints, filtered
to only tensor pairs whose liveness windows (or, under free_schedule, ASAP/
ALAP bounds -- see _asap_alap_tensor_windows()) actually overlap. With
free_schedule=True, it additionally decides the operator schedule itself
(C[a,t]).

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

    Deduplicates activation_inputs/outputs before summing -- an operator
    that reads the same tensor as more than one of its own operands (e.g.
    a self-multiply, Multiply()([x, x]), which TFLite/graph_builder.py
    represents as activation_inputs=[x, x]) only needs that tensor
    resident once in real hardware, not once per occurrence in the list.
    Found via a real model that tripped it: a naive sum() double-counted
    the shared operand, inflating M_R past MPMF -- impossible in
    principle (a node's own inputs+outputs are always a subset of
    whatever _liveness_windows() already counts as resident at that same
    timestep, so M_R can never legitimately exceed MPMF).
    """
    floor_bytes, floor_t = 0, None
    for t in sorted(nodes.keys()):
        node = nodes[t]
        live = sum(tensors[a].size_bytes for a in set(node.activation_inputs) if a in tensors)
        live += sum(tensors[a].size_bytes for a in set(node.outputs) if a in tensors)
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


def build_mpmf_schedule_model(nodes, tensors):
    """
    The paper's §III-E1/Eq.15: find the true, schedule-optimal minimum peak
    memory footprint (M_P) by letting the operator schedule itself --
    C[a,t], which timestep tensor a's producing node runs at -- become a
    free decision, instead of the constant build_cosma_model() fixes it to
    (t == producer_layer(a)). This is what turns the paper's other two
    budgets, M_P and M_H = (M_R + M_P) / 2, from unimplementable into real
    numbers -- see docs/results_plan.md.

    Deliberately excludes everything build_cosma_model() has for memory
    *allocation*: no L (address), no Eq.9 (budget-fit)/Eq.10 (non-overlap)/
    Eq.11 (address-pinning), and no S/R variables at all -- Eq.13 forces
    them to 0 identically in this mode, so they're substituted out of
    Eq.1/2/5 rather than created and then constrained. The paper's own
    text: "memory allocation is not considered" in this mode. Result: a
    small ILP over just C/P (each |A|x|T| binary) plus one scalar M_peak --
    not build_cosma_model()'s full quadratic-pairs machinery.

    Equations used (general form -- NOT the fixed-schedule-simplified
    versions in build_cosma_model(), since C is genuinely free here):
      Eq.1' C[a,t] + P[a,t] <= 1
      Eq.2' P[a,t] <= C[a,t-1] + P[a,t-1]                (base case at T[0]: P==0)
      Eq.5' C[a,t] <= P[b,t]           for every b in a's producing node's
                                        activation_inputs, for every t
      Eq.6  C[a,t] == C[b,t]           for b sharing a's producing node
                                        (siblings created together)
      Eq.7  sum_t C[a,t] == 1          (each tensor created exactly once)
      Eq.14 sum_a (C[a,t]+P[a,t])*size(a) <= M_peak,     for every t
      Eq.15 minimize M_peak

    Plus one constraint NOT in the paper's own numbered list, added here
    after confirming its absence directly against the paper's text: nothing
    in Eq.1-7 as published stops two *different* (non-sibling) tensors from
    both being assigned the same t, even though the paper defines T as
    "each timestep represents the execution of one operator." Enforced
    here as "at most one node's creation per timestep," using one
    representative tensor per node (Eq.6 already ties any siblings
    together, so only one representative is needed per node to avoid
    double-counting them against this same constraint). This follows this
    project's established practice of fixing gaps found in the paper's own
    published formulation rather than silently reproducing them (see the
    Eq.2/3 base-case fix above, and helpers/spm_allocator.py's implicit-
    residency-lapse handling).

    Returns (prob, variables, T, A) -- variables has 'C'/'P'/'M_peak' keys
    only (no 'S'/'R'/'L'/'u'/'d'). extract_results() does NOT understand
    this model's shape -- use extract_mpmf_schedule_results().
    """
    T = sorted(nodes.keys())
    A = sorted(tensors.keys())
    size = {a: tensors[a].size_bytes for a in A}
    # a's producing node -- a STRUCTURAL fact (which node produces tensor a),
    # never itself a timestep/position, even though this same field
    # (tensors[a].producer_layer) happens to equal one under
    # build_cosma_model()'s fixed-schedule convention. Here it's used only
    # to look up that node's own inputs/sibling-outputs.
    producer_node = {a: tensors[a].producer_layer for a in A}
    total_bytes = sum(size.values())

    prob = pulp.LpProblem("cosma_mpmf_schedule", pulp.LpMinimize)

    C = {(a, t): pulp.LpVariable(f"Csched_{a}_{t}", cat='Binary') for a in A for t in T}
    P = {(a, t): pulp.LpVariable(f"Psched_{a}_{t}", cat='Binary') for a in A for t in T}
    M_peak = pulp.LpVariable("M_peak", lowBound=0, upBound=total_bytes, cat='Integer')

    t0 = T[0]
    for a in A:
        for t in T:
            # Eq.1'
            prob += C[a, t] + P[a, t] <= 1, f"MEq1_{a}_{t}"
            # Eq.2' -- same base-case reasoning as build_cosma_model()'s
            # Eq.2/3: no t-1 to chain from at the very first timestep.
            if t > t0:
                prob += P[a, t] <= C[a, t - 1] + P[a, t - 1], f"MEq2_{a}_{t}"
            else:
                prob += P[a, t] == 0, f"MEq2_base_{a}_{t}"

    # Eq.5' -- evaluated at EVERY t (not just one fixed producer_t[a], since
    # which t ends up hosting a's creation isn't known until solved):
    # whichever t a's producing node is assigned to (C[a,t]=1), that node's
    # own needed inputs must be resident (P; R is 0 in this mode) at that
    # same t. Vacuous (0 <= anything) for every t where C[a,t]=0.
    for a in A:
        node = nodes[producer_node[a]]
        for b in node.activation_inputs:
            if b in tensors:
                for t in T:
                    prob += C[a, t] <= P[b, t], f"MEq5_{a}_{b}_{t}"

    # Eq.6 -- tensors sharing the same producing node must be created at the
    # same t (vacuous on every model tested so far: none has a multi-output
    # node yet -- see docs/results_plan.md's verification notes).
    for a in A:
        node = nodes[producer_node[a]]
        siblings = [s for s in node.outputs if s != a and s in tensors]
        for b in siblings:
            for t in T:
                prob += C[a, t] == C[b, t], f"MEq6_{a}_{b}_{t}"

    # Eq.7 -- each tensor created exactly once.
    for a in A:
        prob += pulp.lpSum(C[a, t] for t in T) == 1, f"MEq7_{a}"

    # Not in the paper's own numbered equations (see docstring): at most one
    # node's creation per timestep. One representative tensor per node --
    # Eq.6 already forces any siblings to share C together, so using more
    # than one representative would double-count a single multi-output
    # node against itself.
    representative = {}
    for a in A:
        n = producer_node[a]
        representative.setdefault(n, a)
    for t in T:
        prob += (pulp.lpSum(C[rep, t] for rep in representative.values()) <= 1,
                 f"AtMostOneNodePerTimestep_{t}")

    # Eq.14
    for t in T:
        prob += (pulp.lpSum((C[a, t] + P[a, t]) * size[a] for a in A) <= M_peak,
                 f"MEq14_{t}")

    # Eq.15
    prob += M_peak

    variables = {'C': C, 'P': P, 'M_peak': M_peak}
    return prob, variables, T, A


def extract_mpmf_schedule_results(variables, T, A) -> dict:
    """
    Returns {'mpmf_bytes': int, 'schedule': {a: t}} for a solved
    build_mpmf_schedule_model() -- 'schedule' is COSMA's own chosen
    operator order for minimum peak footprint (the t where C[a,t]==1 for
    each tensor a), not necessarily model.json's input order. Not consumed
    by run_cosma.py/baseline.py -- those still simulate against the fixed
    input order regardless (see docs/results_plan.md).
    """
    C, P, M_peak = variables['C'], variables['P'], variables['M_peak']

    def val(x):
        v = pulp.value(x)
        return 0 if v is None else int(round(v))

    schedule = {}
    for a in A:
        for t in T:
            if val(C[a, t]):
                schedule[a] = t
                break

    return {'mpmf_bytes': val(M_peak), 'schedule': schedule}


def compute_true_mpmf_bytes(nodes, tensors, time_limit_sec=None, solver='cbc') -> Tuple[int, dict]:
    """
    The paper's real M_P (not compute_mpmf_bytes()'s fixed-schedule proxy):
    build_mpmf_schedule_model() + solve() + extract_mpmf_schedule_results()
    in one call, matching compute_structural_minimum_bytes()/
    compute_mpmf_bytes()'s simple call shape. Unlike those two, this pays
    for a real ILP solve -- not instant, can be slow on a large model (same
    "C becomes a full |T|x|A| binary block" jump in solve difficulty as
    full operator rescheduling generally -- see docs/ITERATION_HISTORY.md).

    Returns (bytes, schedule). Raises RuntimeError on a non-Optimal status,
    same pattern as run_cosma.py's main-pipeline solve.

    time_limit_sec: None (default) means unbounded. If you do pass a limit
    and get back a non-Optimal status (including 'Infeasible'), don't
    trust it as a proof: CBC/PuLP's status reporting under a time limit is
    not reliable, and a solve that merely ran out of time can come back
    labeled 'Infeasible' rather than 'Not Solved'. Confirmed directly on a
    real model (_exported/fake2/) -- reported 'Infeasible' at the default
    120s CLI limit despite the graph's own execution order being a
    hand-verified, zero-constraint-violation feasible solution to this
    exact model. See visualize_spm.py's solve_ilp_only() docstring for the
    same caveat on the fixed-schedule solve.
    """
    prob, variables, T, A = build_mpmf_schedule_model(nodes, tensors)
    status = solve(prob, time_limit_sec=time_limit_sec, solver=solver)
    if status != 'Optimal':
        # The CBC/PuLP mislabeling risk below is specific to CBC's own
        # status parsing -- Gurobi's status codes reliably distinguish
        # "time limit reached" from "proven infeasible", so this caveat
        # doesn't apply when solver='gurobi'.
        caveat = (
            f" -- a time limit ({time_limit_sec}s) was set, so this status is NOT "
            f"necessarily a proof: a solve that simply didn't finish in time can come "
            f"back mislabeled 'Infeasible' rather than 'Not Solved'. Retry with a "
            f"longer time_limit_sec or None (unbounded) before trusting this as a "
            f"real infeasibility."
            if time_limit_sec and solver == 'cbc' else ""
        )
        raise RuntimeError(f"MPMF schedule ILP did not solve to optimality: "
                            f"status={status}{caveat}")
    result = extract_mpmf_schedule_results(variables, T, A)
    return result['mpmf_bytes'], result['schedule']


def _asap_alap_node_windows(nodes, tensors, T) -> Dict[int, Tuple[int, int]]:
    """
    Provably-correct outer bound on which timestep each node could ever run
    at, under ANY valid topological order -- used to safely prune Eq.10's
    u/d pairs when free_schedule=True. The fixed-schedule pair-filter
    (_liveness_windows(), producer_layer/consumer_layers read as literal
    timesteps) is meaningless once C is free -- see build_cosma_model()'s
    free_schedule docstring.

    asap[n]: longest path from any source (a node with no activation
    inputs) to n, counting nodes -- the earliest timestep n could run.
    alap[n]: the symmetric bound from the sinks -- the latest timestep n
    could run without forcing some later node out of room. Both computed
    in one forward + one backward pass, O(N + E), using each node's
    activation_inputs' tensors[b].producer_layer as the DAG's edges (the
    same producer lookup graph_builder.py already computed -- no need to
    rescan every node's outputs to find it).

    Iterates node ids in sorted order for the forward pass (and reversed
    for the backward pass) -- valid because sorted(nodes.keys()) is
    already a real topological order of this DAG (the same fact
    build_cosma_model's fixed-schedule mode already relies on: model.json
    layer ids are assigned in topological order -- see this module's own
    docstring).
    """
    node_ids = sorted(nodes.keys())
    N = len(T)

    preds: Dict[int, List[int]] = {n: [] for n in node_ids}
    succs: Dict[int, List[int]] = {n: [] for n in node_ids}
    for n in node_ids:
        for b in nodes[n].activation_inputs:
            if b in tensors:
                producer = tensors[b].producer_layer
                preds[n].append(producer)
                succs[producer].append(n)

    asap: Dict[int, int] = {}
    for n in node_ids:
        asap[n] = 0 if not preds[n] else 1 + max(asap[p] for p in preds[n])

    alap: Dict[int, int] = {}
    for n in reversed(node_ids):
        alap[n] = (N - 1) if not succs[n] else min(alap[s] for s in succs[n]) - 1

    return {n: (asap[n], alap[n]) for n in node_ids}


def _asap_alap_tensor_windows(tensors, node_windows: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
    """
    Tensor version of _asap_alap_node_windows(): a tensor can't exist
    before its producer's earliest possible timestep (asap), and the
    latest it could still be needed is the latest any of its consumers
    (or, absent any, its own producer) could possibly run (alap) -- a
    safe superset of the real window, never tighter than what the actual
    solve could produce, so pruning on it never drops a pair that could
    truly overlap.
    """
    windows = {}
    for a, t in tensors.items():
        asap_p, alap_p = node_windows[t.producer_layer]
        end_candidates = [alap_p] + [node_windows[c][1] for c in t.consumer_layers
                                      if c in node_windows]
        windows[a] = (asap_p, max(end_candidates))
    return windows


def build_cosma_model(nodes, tensors, memory_budget_bytes: int, free_schedule: bool = False):
    """
    nodes:   dict[int, graph_builder.Node]
    tensors: dict[int, graph_builder.Tensor]  (COSMA-tracked activations only)
    memory_budget_bytes: SPM capacity, also used as the big-M constant
        (per the plan doc's Eq.10 risk mitigation: "Set M = memory_budget,
        not a huge arbitrary number").
    free_schedule: False (default) keeps this module's original behavior --
        C[a,t] ("tensor a created at t") fixed to model.json's topological
        order, byte-identical to every previously validated result. True
        turns C into a real decision (the paper's actual "Combined
        Scheduling..." headline, §III-B/C): the ILP is now free to choose
        which node runs at which timestep, not just how to place/replace
        tensors given a fixed order. This restores Eq.6 (siblings created
        together) and Eq.7 (each tensor created exactly once) as real
        constraints -- true only by construction under the fixed schedule
        -- adds an explicit "at most one node's creation per timestep"
        constraint (which, combined with Eq.7 and |T| == |nodes|, forces
        an exact bijection by pigeonhole -- no separate "exactly one node"
        constraint is needed), and evaluates Eq.5 at every t instead of
        just the one fixed producer timestep. DAG precedence between a
        node and its input-producing nodes is NOT separately encoded --
        it's implied transitively through Eq.1/2/5 (a node's inputs must
        be P/R-resident at its own creation instant, which requires their
        own producer's C to have fired strictly earlier) -- the same
        argument already used (and empirically verified) for
        build_mpmf_schedule_model(); re-verified here too, see
        docs/ITERATION_HISTORY.md and the test scripts run alongside this
        change. Because Eq.10's pair-filter can no longer assume
        producer_layer/consumer_layers are literal timesteps once C is
        free, it switches to _asap_alap_tensor_windows() -- a provably
        correct (if looser) outer bound. Expect solve time to jump
        substantially: this is the paper's own documented O(|T|x|A|^2)
        worst case.

    Returns (prob, variables, T, A) where variables is a dict of the
    PuLP variable dicts (P, S, R, L, u, d, C). C is a constant-returning
    closure when free_schedule=False (as before), or a dict of
    LpVariables (keyed (a, t)) when free_schedule=True -- see
    extract_results()'s callable(C) dispatch.
    """
    T = sorted(nodes.keys())
    A = sorted(tensors.keys())
    budget = memory_budget_bytes
    size = {a: tensors[a].size_bytes for a in A}
    producer_t = {a: tensors[a].producer_layer for a in A}

    assert_tensors_fit_budget(tensors, budget)

    prob = pulp.LpProblem("COSMA", pulp.LpMinimize)

    if free_schedule:
        C = {(a, t): pulp.LpVariable(f"C_{a}_{t}", cat='Binary') for a in A for t in T}

        def Cv(a, t):
            return C[a, t]
    else:
        def C(a, t):
            return 1 if t == producer_t[a] else 0
        Cv = C

    P = {(a, t): pulp.LpVariable(f"P_{a}_{t}", cat='Binary') for a in A for t in T}
    S = {(a, t): pulp.LpVariable(f"S_{a}_{t}", cat='Binary') for a in A for t in T}
    R = {(a, t): pulp.LpVariable(f"R_{a}_{t}", cat='Binary') for a in A for t in T}
    L = {(a, t): pulp.LpVariable(f"L_{a}_{t}", lowBound=0, upBound=budget, cat='Integer')
         for a in A for t in T}

    def resident(a, t):
        return Cv(a, t) + P[a, t] + R[a, t]

    t0 = T[0]

    # Eq.1: at most one action per tensor per timestep
    for a in A:
        for t in T:
            prob += Cv(a, t) + P[a, t] + S[a, t] + R[a, t] <= 1, f"Eq1_{a}_{t}"

    # Eq.2/3: preserve/spill only if resident at t-1; Eq.4: retrieve only if spilled by t.
    # Base case (t == T[0]): there is no t-1 for Eq.2/3 to chain from, so
    # nothing can be "already resident" yet -- without this, P[a,T[0]]/
    # S[a,T[0]] are left completely unconstrained for every tensor (Eq.1
    # alone doesn't forbid them), letting the solver plant a zero-cost
    # "phantom" P (or, worse, a genuinely double-counted S) on a tensor
    # before it's even created. Confirmed to actually happen on ResNet-50
    # (5 of 79 tensors got a spurious P at t=0 in an optimal solve) --
    # silent on every previously-validated smaller model, but a real gap.
    for a in A:
        for t in T:
            if t > t0:
                prob += P[a, t] <= Cv(a, t - 1) + P[a, t - 1] + R[a, t - 1], f"Eq2_{a}_{t}"
                prob += S[a, t] <= Cv(a, t - 1) + P[a, t - 1], f"Eq3_{a}_{t}"
            else:
                prob += P[a, t] == 0, f"Eq2_base_{a}_{t}"
                prob += S[a, t] == 0, f"Eq3_base_{a}_{t}"
            prob += R[a, t] <= pulp.lpSum(S[a, k] for k in T if k <= t), f"Eq4_{a}_{t}"

    # Eq.5: an operator's activation inputs must already be resident when it runs.
    if free_schedule:
        # C is free, so which t ends up hosting a's creation isn't known
        # until solved -- evaluated at every t (vacuous whenever
        # Cv(a,t)==0). This is also what makes DAG precedence hold
        # transitively -- see the free_schedule docstring above.
        for a in A:
            node = nodes[producer_t[a]]
            for b in node.activation_inputs:
                if b in tensors:
                    for t in T:
                        prob += Cv(a, t) <= P[b, t] + R[b, t], f"Eq5_{a}_{b}_{t}"
    else:
        # Schedule is fixed, so this only needs checking at t = producer_t[a].
        for a in A:
            t = producer_t[a]
            node = nodes[t]
            for b in node.activation_inputs:
                if b in tensors:
                    prob += Cv(a, t) <= P[b, t] + R[b, t], f"Eq5_{a}_{b}_{t}"

    if free_schedule:
        # Eq.6: tensors sharing the same producing node must be created at
        # the same t -- a real constraint now that t is a decision.
        for a in A:
            node = nodes[producer_t[a]]
            siblings = [s for s in node.outputs if s != a and s in tensors]
            for b in siblings:
                for t in T:
                    prob += Cv(a, t) == Cv(b, t), f"Eq6_{a}_{b}_{t}"

        # Eq.7: each tensor created exactly once.
        for a in A:
            prob += pulp.lpSum(Cv(a, t) for t in T) == 1, f"Eq7_{a}"

        # Not one of the paper's own numbered equations: at most one
        # node's creation per timestep, using one representative tensor
        # per node (Eq.6 already ties any siblings together). Combined
        # with Eq.7 and |T| == |nodes|, this forces an exact bijection by
        # pigeonhole -- no separate "exactly one node per timestep"
        # constraint needed. Same fix already applied to
        # build_mpmf_schedule_model() after confirming its absence
        # directly against the paper's own text.
        representative: Dict[int, int] = {}
        for a in A:
            representative.setdefault(producer_t[a], a)
        for t in T:
            prob += (pulp.lpSum(Cv(rep, t) for rep in representative.values()) <= 1,
                     f"AtMostOneNodePerTimestep_{t}")
    # else: Eq.6 and Eq.7 hold by construction (see module docstring).

    # Eq.8: each tensor spilled at most once
    for a in A:
        prob += pulp.lpSum(S[a, t] for t in T) <= 1, f"Eq8_{a}"

    # Eq.9: resident tensor must fit the budget (relaxed to a no-op when not resident)
    for a in A:
        for t in T:
            prob += (L[a, t] + size[a] <= budget + budget * (1 - resident(a, t)),
                     f"Eq9_{a}_{t}")

    # Overlap-filtered tensor pairs for Eq.10 (the quadratic-blowup mitigation).
    # Fixed schedule: producer_layer/consumer_layers ARE literal timesteps,
    # so the exact liveness window applies. Free schedule: neither is known
    # until solved, so a provably-correct (safe superset) ASAP/ALAP bound
    # is used instead -- see _asap_alap_tensor_windows()'s docstring.
    if free_schedule:
        windows = _asap_alap_tensor_windows(tensors, _asap_alap_node_windows(nodes, tensors, T))
    else:
        windows = _liveness_windows(tensors)

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
            if t > t0:
                prob += (L[a, t] - L[a, t - 1] <= budget * (1 - V[a, t]),
                         f"Eq11a_{a}_{t}")
                prob += (L[a, t - 1] - L[a, t] <= budget * (1 - V[a, t]),
                         f"Eq11b_{a}_{t}")

    # Eq.12: minimize non-compulsory (spill + retrieve) DRAM traffic.
    prob += pulp.lpSum((S[a, t] + R[a, t]) * size[a] for a in A for t in T)

    variables = {'P': P, 'S': S, 'R': R, 'L': L, 'u': u, 'd': d, 'C': C}
    return prob, variables, T, A


def solve(prob, time_limit_sec=None, msg=False, solver='cbc'):
    """
    solver: 'cbc' (default -- PuLP's bundled open-source solver, no license
        needed, always available) or 'gurobi' (commercial, requires a
        working license -- see docs/STATUS.md's solver gap note: CBC
        measured ~600x slower than Gurobi on Inception-V3-sized problems,
        178s vs. the paper's own 0.296s average). Uses pulp.GUROBI (the
        in-process gurobipy binding, not GUROBI_CMD, which needs the
        separate command-line binary we haven't installed) -- raises
        pulp.PulpSolverError with a clear message if gurobipy isn't
        installed or no license activates, rather than silently falling
        back to CBC (a silent fallback would hide exactly the solver
        identity this project has been careful to always disclose
        alongside every solve-time number).
    """
    if solver == 'gurobi':
        pulp_solver = pulp.GUROBI(msg=msg, timeLimit=time_limit_sec)
    elif solver == 'cbc':
        pulp_solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec)
    else:
        raise ValueError(f"Unknown solver {solver!r} -- expected 'cbc' or 'gurobi'")
    prob.solve(pulp_solver)
    return pulp.LpStatus[prob.status]


def extract_results(variables, T, A, tensors):
    """
    Returns:
      ordered_layers:      layer execution order -- model.json's fixed
                            topological order when the solve came from
                            free_schedule=False, or the ILP's own chosen
                            order when free_schedule=True (see
                            schedule_layer_at_t below).
      schedule_layer_at_t: timestep -> layer id actually running there.
                            Identity ({t: t}) under a fixed schedule (t IS
                            the layer id); under a free schedule, derived
                            from which tensor's C[a,t]==1 at each t. This
                            is the mapping downstream code (baseline.py's
                            run_cosma_aware(), run_cosma.py's per-timestep
                            loop) needs to look up the right node/layer for
                            an abstract timestep t once t no longer equals
                            the layer id.
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

    # C is a constant-returning closure under a fixed schedule, or a dict
    # of LpVariables under a free one (see build_cosma_model()'s
    # free_schedule docstring) -- this dispatches on which.
    def c_val(a, t):
        return C(a, t) if callable(C) else val(C[a, t])

    if callable(C):
        schedule_layer_at_t = {t: t for t in T}
    else:
        producer_node = {a: tensors[a].producer_layer for a in A}
        schedule_layer_at_t = {}
        for t in T:
            for a in A:
                if c_val(a, t):
                    schedule_layer_at_t[t] = producer_node[a]
                    break
        missing = [t for t in T if t not in schedule_layer_at_t]
        assert not missing, (
            f"free-schedule ILP solved but timestep(s) {missing} have no "
            f"node assigned -- Eq.7 + 'at most one node per timestep' "
            f"should make this impossible for an Optimal solve; a real "
            f"bug in the ILP or this extraction if it ever fires")

    ordered_layers = [schedule_layer_at_t[t] for t in T]

    extra_dram_bytes = {t: 0 for t in T}
    resident_action = {}
    for a in A:
        size = tensors[a].size_bytes
        for t in T:
            if c_val(a, t):
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
            if c_val(a, t) or val(P[a, t]) or val(R[a, t]):
                spm_plan[(a, t)] = val(L[a, t])

    return {
        'ordered_layers': ordered_layers,
        'schedule_layer_at_t': schedule_layer_at_t,
        'extra_dram_bytes': extra_dram_bytes,
        'spm_plan': spm_plan,
        'resident_action': resident_action,
    }
