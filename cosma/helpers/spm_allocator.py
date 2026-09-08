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
