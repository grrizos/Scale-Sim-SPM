# cosma/helpers/spm_allocator.py
"""
A real, live, byte-addressed SPM occupancy tracker that replays COSMA's
solved plan (resident_action/spm_plan from cosma_Ilp.extract_results())
one timestep at a time, verifying every Create/Preserve/Spill/Retrieve
transition is physically consistent (no double-booking, no freeing a
tensor that was never there, no address collision, never over budget).

Why this exists: baseline.py's real SCALE-Sim loop only ever reads
resident_action's per-tensor 'P' flag to decide whether to skip a single
tensor's own fetch -- it has no live, cross-layer bookkeeping of its own,
so nothing would ever catch a genuine capacity violation or address
collision if the ILP's plan and the engine's execution ever disagreed.
This module is that missing, independent check -- it doesn't change what
gets simulated, it verifies that what gets simulated is physically
realizable at the declared budget.

Standalone: no scalesim/numpy imports, so it can run two ways -- driven
live from inside baseline.py's real per-layer loop (see
_run_layers()/run_cosma_aware()), or driven standalone against a solved
ILP result with zero SCALE-Sim involvement (needed for
toy_spill_model.json/toy_branching_model.json, which have no real conv
params and can't run through baseline.py/topology_builder.py at all --
see those files' own _comment fields).

Two non-obvious things confirmed by actually solving toy_spill_model.json
(budget=200) and reading the real resident_action/spm_plan back, not
just reasoned about on paper -- both required real fixes to get right:

1. **Ordering**: a tensor can be Spilled at the same timestep another
   tensor is Created/Retrieved into exactly the address space just
   freed. In the real solve: tensor 10 (10B) is Spilled at t=1, freeing
   address 0, and tensor 11 (100B) is Created at that same t=1, address
   0. step() always processes every Spill (and every implicit lapse,
   below) before any Create/Retrieve at that same timestep, or this
   exact case raises a false collision.
2. **Implicit lapse, not just explicit Spill**: a tensor whose last
   consumer has already run can simply stop appearing in
   resident_action -- no explicit 'S' at all -- once nothing ever
   retrieves it again. This is optimal ILP behavior, not a gap: Eq.12's
   objective charges real bytes for an explicit Spill (even one never
   retrieved), but charges nothing for just letting P end, so the
   solver has no reason to ever mark 'S' for a tensor it will never
   retrieve. In the real solve, tensor 11 is 'P' at t=2 and then has
   *no* entry at all at t=3 -- yet tensor 10's Retrieve at t=3 correctly
   reuses tensor 11's old address 0, which is only safe because tensor
   11 has, in fact, vacated. step() frees anything no longer in
   spm_plan's claimed resident set for t, whether or not an explicit
   'S' was ever recorded for it -- treating only explicit 'S' as "freed"
   (an earlier, wrong version of this module did exactly that) produces
   a false collision here.
"""
from typing import Dict, List, Tuple


class SpmAllocationError(RuntimeError):
    """
    Raised when a solved COSMA plan's claimed transition isn't physically
    realizable in a live byte-addressed replay -- e.g. two tensors
    claiming overlapping addresses, or a Preserve/Spill for a tensor the
    replay never saw allocated. Should be unreachable for an Optimal ILP
    solve (Eq.9/10 guarantee exactly this); if it ever fires, that's a
    real bug in the ILP, the extraction step, or a timestep-semantics
    mismatch -- not a tuning issue.
    """
    def __init__(self, message: str, *, tensor_id: int = None,
                 timestep: int = None, reason: str = None):
        super().__init__(message)
        self.tensor_id = tensor_id
        self.timestep = timestep
        self.reason = reason


class SpmAllocator:
    """
    Live replay of a solved COSMA plan against a fixed-size byte-addressed
    SPM. Call step(t) once per timestep, in strictly increasing order (or
    replay_all() to do every timestep at once) -- see module docstring.
    """

    def __init__(self, tensors: Dict[int, object],
                 spm_plan: Dict[Tuple[int, int], int],
                 resident_action: Dict[Tuple[int, int], str],
                 memory_budget_bytes: int):
        self._tensors = tensors
        self._spm_plan = spm_plan
        self._budget = memory_budget_bytes
        self._occupants: Dict[int, Tuple[int, int]] = {}  # tensor_id -> (address, size)
        self._last_t = None
        self._peak_bytes = 0
        self._steps_taken = 0

        # Grouped once, O(N) total, rather than rescanning the full dict
        # on every step().
        self._actions_by_t: Dict[int, List[Tuple[int, str]]] = {}
        for (a, t), action in resident_action.items():
            self._actions_by_t.setdefault(t, []).append((a, action))

        self._plan_resident_by_t: Dict[int, set] = {}
        for (a, t) in spm_plan:
            self._plan_resident_by_t.setdefault(t, set()).add(a)

    def step(self, t: int) -> None:
        """
        Replay every action at timestep t against the live SPM state.
        Must be called with strictly increasing t (replay_all() handles
        this automatically). Processes all Spills first, then all
        Create/Retrieve allocations, then verifies all Preserves -- see
        module docstring for why that order is required. Ends by
        cross-checking the resulting live-occupant set against
        spm_plan's own claimed resident set for t -- a genuine,
        non-tautological check, since resident_action and spm_plan are
        built by two separate loops over the same solved ILP variables
        in cosma_Ilp.extract_results().
        """
        if self._last_t is not None and t <= self._last_t:
            raise SpmAllocationError(
                f"step() called out of order: t={t} after t={self._last_t}",
                timestep=t, reason='out_of_order')
        self._last_t = t

        actions = self._actions_by_t.get(t, [])
        expected = self._plan_resident_by_t.get(t, set())
        before = set(self._occupants.keys())

        # Explicit spills must actually have been resident.
        for a, action in actions:
            if action == 'S':
                self._free(a, t)

        # Free everything no longer claimed resident at t -- an explicit
        # Spill (just validated above) or an *implicit* lapse: the ILP
        # has no obligation to ever mark a tensor 'S' if it's simply
        # never retrieved again (letting P end costs nothing in Eq.12's
        # objective, unlike a real Spill, which costs its bytes even if
        # never retrieved -- so the solver's natural, optimal choice is
        # to just let residency lapse for free). Both cases free the
        # same way; this single set-difference handles them uniformly,
        # and must run before any Create/Retrieve allocation below --
        # see toy_spill_model.json's tensor 11 (lapses after t=2, no
        # explicit 'S' at all) freeing address 0 for tensor 10's t=3
        # retrieve into that same address.
        for a in before - expected:
            del self._occupants[a]

        for a, action in actions:
            if action in ('C', 'R'):
                self._allocate(a, t, action)
        for a, action in actions:
            if action == 'P':
                self._preserve(a, t)

        actual = set(self._occupants.keys())
        if actual != expected:
            missing = expected - actual
            extra = actual - expected
            raise SpmAllocationError(
                f"live occupancy at t={t} disagrees with spm_plan's claimed "
                f"resident set (missing={sorted(missing)}, extra={sorted(extra)})",
                timestep=t, reason='snapshot_mismatch')

        self._steps_taken += 1
        self._peak_bytes = max(self._peak_bytes, self.occupied_bytes())

    def replay_all(self, timesteps: List[int] = None) -> None:
        """
        Standalone use, no baseline.py/SCALE-Sim involved -- calls step(t)
        for every t in order. This is what toy fixture verification uses
        (see toy_spill_model.json/toy_branching_model.json, which can't
        run through baseline.py at all).
        """
        if timesteps is None:
            timesteps = sorted(set(self._actions_by_t) | set(self._plan_resident_by_t))
        for t in timesteps:
            self.step(t)

    def _allocate(self, tensor_id: int, t: int, action: str) -> None:
        if tensor_id in self._occupants:
            raise SpmAllocationError(
                f"tensor {tensor_id} action '{action}' at t={t} but it's "
                f"already tracked resident at {self._occupants[tensor_id]}",
                tensor_id=tensor_id, timestep=t, reason='double_allocate')

        if (tensor_id, t) not in self._spm_plan:
            raise SpmAllocationError(
                f"tensor {tensor_id} action '{action}' at t={t} has no "
                f"spm_plan address entry",
                tensor_id=tensor_id, timestep=t, reason='missing_plan_entry')

        address = self._spm_plan[(tensor_id, t)]
        size = self._tensors[tensor_id].size_bytes

        if address < 0 or address + size > self._budget:
            raise SpmAllocationError(
                f"tensor {tensor_id} action '{action}' at t={t}: "
                f"[{address}, {address + size}) exceeds budget {self._budget}",
                tensor_id=tensor_id, timestep=t,
                reason='address_range_exceeds_budget')

        for other_id, (other_addr, other_size) in self._occupants.items():
            if address < other_addr + other_size and other_addr < address + size:
                raise SpmAllocationError(
                    f"tensor {tensor_id} action '{action}' at t={t}: "
                    f"[{address}, {address + size}) collides with tensor "
                    f"{other_id} at [{other_addr}, {other_addr + other_size}) -- "
                    f"live occupants: {self._occupant_dump()}",
                    tensor_id=tensor_id, timestep=t, reason='collision')

        new_total = sum(s for _, s in self._occupants.values()) + size
        if new_total > self._budget:
            raise SpmAllocationError(
                f"tensor {tensor_id} action '{action}' at t={t}: total "
                f"occupied {new_total} exceeds budget {self._budget}",
                tensor_id=tensor_id, timestep=t, reason='capacity_exceeded')

        self._occupants[tensor_id] = (address, size)

    def _free(self, tensor_id: int, t: int) -> None:
        """
        Validates an explicit Spill claim only -- the actual removal from
        self._occupants happens uniformly in step() (see there), together
        with any tensor whose residency implicitly lapses the same way.
        """
        if tensor_id not in self._occupants:
            raise SpmAllocationError(
                f"tensor {tensor_id} action 'S' at t={t} but it isn't "
                f"currently tracked resident",
                tensor_id=tensor_id, timestep=t, reason='free_without_residency')

    def _preserve(self, tensor_id: int, t: int) -> None:
        if tensor_id not in self._occupants:
            raise SpmAllocationError(
                f"tensor {tensor_id} action 'P' at t={t} but it isn't "
                f"currently tracked resident",
                tensor_id=tensor_id, timestep=t, reason='preserve_without_residency')

        address, _ = self._occupants[tensor_id]
        plan_address = self._spm_plan.get((tensor_id, t))
        if plan_address != address:
            raise SpmAllocationError(
                f"tensor {tensor_id} action 'P' at t={t}: live address "
                f"{address} disagrees with spm_plan's {plan_address}",
                tensor_id=tensor_id, timestep=t, reason='preserve_address_mismatch')

    def _occupant_dump(self) -> str:
        return ', '.join(
            f"{tid}@[{addr},{addr + size})"
            for tid, (addr, size) in sorted(self._occupants.items()))

    def is_resident(self, tensor_id: int) -> bool:
        return tensor_id in self._occupants

    def address_of(self, tensor_id: int):
        entry = self._occupants.get(tensor_id)
        return entry[0] if entry else None

    def occupied_bytes(self) -> int:
        return sum(size for _, size in self._occupants.values())

    def peak_occupied_bytes(self) -> int:
        return self._peak_bytes

    def steps_taken(self) -> int:
        return self._steps_taken


def _lowest_fit_address(occupied: List[Tuple[int, int]], size: int,
                         budget: int) -> int:
    """
    Lowest free address in [0, budget) with `size` free contiguous bytes,
    given `occupied` ((address, size) pairs) already placed -- classic
    lowest-fit interval packing: scan gaps left to right, return the
    first one that fits. Raises SpmAllocationError if none does.
    """
    cursor = 0
    for address, occ_size in sorted(occupied):
        if address - cursor >= size:
            return cursor
        cursor = max(cursor, address + occ_size)
    if budget - cursor >= size:
        return cursor
    raise SpmAllocationError(
        f"compact_spm_plan(): no {size}-byte gap fits in a {budget}-byte "
        f"budget against current occupants {sorted(occupied)} -- a real "
        f"dynamic-storage-allocation heuristic failure (fragmentation),"
        f"not necessarily a bug -- see compact_spm_plan()'s docstring; "
        f"the caller should fall back to the original spm_plan.",
        reason='compaction_overflow')


def _residency_episodes(tensors: Dict[int, object],
                         resident_action: Dict[Tuple[int, int], str]
                         ) -> List[Tuple[int, int, int]]:
    """
    Maximal contiguous (by consecutive integer t) runs of C/P/R residency
    per tensor -- an "episode" needs one stable address for its whole
    span (Eq.11's own address-pinning semantics: a tensor's address only
    has to stay fixed while it's continuously resident). A tensor
    spilled and later retrieved (Eq.8: at most once) gets two
    independent episodes, each placeable at a different address, exactly
    like a real solve already allows.

    Returns a list of (tensor_id, t_start, t_end), t_end inclusive.
    """
    ts_by_tensor: Dict[int, List[int]] = {}
    for (a, t), action in resident_action.items():
        if action in ('C', 'P', 'R'):
            ts_by_tensor.setdefault(a, []).append(t)

    episodes: List[Tuple[int, int, int]] = []
    for a, ts in ts_by_tensor.items():
        ts.sort()
        start = prev = ts[0]
        for t in ts[1:]:
            if t == prev + 1:
                prev = t
            else:
                episodes.append((a, start, prev))
                start = prev = t
        episodes.append((a, start, prev))
    return episodes


def compact_spm_plan(tensors: Dict[int, object],
                      resident_action: Dict[Tuple[int, int], str],
                      memory_budget_bytes: int) -> Dict[Tuple[int, int], int]:
    """
    Re-addresses a solved COSMA plan for VISUALIZATION only -- same
    resident_action (which tensor is C/P/R/S at which timestep, never
    touched), but repacked toward address 0 instead of whatever arbitrary
    addresses the ILP's own `L` variable happened to land on.

    Why this exists: cosma_Ilp.py's Eq.9-11 only ever constrain `L` to fit
    the budget and not overlap -- nothing in Eq.12's objective involves
    `L` at all (verified against the paper's own text: see
    docs/STATUS.md's "nothing rewards compact placement" note), so CBC
    returns any feasible address assignment with zero preference for
    compactness. Confirmed directly on a real solve (ResNet-20-CIFAR10 @
    256KB): a single resident tensor at t=0, with nothing else resident
    and the whole budget free, landed at address 65536 -- address 0 sat
    empty for no reason. The resulting plot shows scattered boxes with
    pointless gaps, technically correct but confusing to read.

    Algorithm: offline, size-first interval placement, not a timestep-by-
    timestep online greedy. Since resident_action is known for the whole
    horizon up front (this is not actually a streaming problem), each
    tensor's full C/P/R lifetime is first grouped into "episodes"
    (_residency_episodes() -- a stable-address span, per Eq.11), then
    episodes are placed largest-first (ties broken by start time, then
    tensor id, for a deterministic result), each at the lowest address
    free for its *entire* span against every already-placed episode it
    overlaps in time. This was NOT the first design tried here: an
    earlier timestep-by-timestep online greedy (commit history) failed
    outright on the small custom DenseNet fixture's real spill/retrieve
    plan (two 160KB tensors placed with a 64KB gap between them, too
    fragmented for a later 192KB tensor to fit even though 235KB of free
    space existed in total) -- placing large, long-lived tensors first
    avoids exactly that failure mode by giving them first pick of
    contiguous space before smaller/shorter-lived ones can wedge into it.

    A repacking is always feasible at the SAME budget in principle -- it
    changes only WHERE each tensor sits, never which tensors are resident
    when (resident_action, held fixed) or how many bytes are
    simultaneously occupied at any timestep (a property of
    resident_action/tensor sizes alone, and Eq.9 already guarantees that
    total never exceeds budget). In practice, general dynamic storage
    allocation (variable-size objects, live ranges known in advance) is
    NP-hard -- the same reason the real placement needs an ILP at all --
    so even this smarter heuristic is not a proof it will always succeed;
    the result is self-verified below rather than assumed, and the
    caller should fall back to the original spm_plan on
    SpmAllocationError (see visualize_spm.py).

    Returns a new spm_plan dict, (tensor_id, t) -> address, the same shape
    cosma_Ilp.extract_results()['spm_plan'] has -- a drop-in replacement
    for rendering. Self-verified before returning by replaying the result
    back through a real SpmAllocator against the same resident_action --
    raises SpmAllocationError (not a silent wrong answer) if that replay
    ever disagrees, matching this module's own "verify, don't trust"
    practice for everything else it does.
    """
    episodes = _residency_episodes(tensors, resident_action)
    episodes.sort(key=lambda e: (-tensors[e[0]].size_bytes, e[1], e[0]))

    placed: List[Tuple[int, int, int, int]] = []  # (address, size, t_start, t_end)
    address_by_episode: Dict[Tuple[int, int], int] = {}  # (tensor_id, t_start) -> address

    for (a, t_start, t_end) in episodes:
        size = tensors[a].size_bytes
        conflicting = [(addr, sz) for (addr, sz, s, e) in placed
                       if s <= t_end and t_start <= e]
        address = _lowest_fit_address(conflicting, size, memory_budget_bytes)
        placed.append((address, size, t_start, t_end))
        address_by_episode[(a, t_start)] = address

    episode_start_of: Dict[Tuple[int, int], int] = {}
    for (a, t_start, t_end) in episodes:
        for t in range(t_start, t_end + 1):
            episode_start_of[(a, t)] = t_start

    spm_plan: Dict[Tuple[int, int], int] = {}
    for (a, t), action in resident_action.items():
        if action in ('C', 'P', 'R'):
            spm_plan[(a, t)] = address_by_episode[(a, episode_start_of[(a, t)])]

    verifier = SpmAllocator(tensors, spm_plan, resident_action, memory_budget_bytes)
    verifier.replay_all()

    return spm_plan
