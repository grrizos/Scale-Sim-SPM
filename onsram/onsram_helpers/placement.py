# onsram/onsram_helpers/placement.py
"""
OnSRAM's own WHERE-placement step: turns resident_action (WHAT/WHEN,
already decided by pinning.py) into concrete SPM byte addresses. This is
OnSRAM's own algorithm, not a borrowed one -- OnSRAM's own paper never
specifies (or needs) a byte-address placement scheme at all; its own
evaluation only ever tracked a `pinned: bool` per tensor. Needing actual
addresses at all is purely a consequence of wiring into real SCALE-Sim
(via a live SpmAllocator replay, which needs a concrete address to check
against), so this module is free to use whatever placement strategy suits
OnSRAM's own decision patterns best, rather than inheriting a heuristic
tuned for a different system's typical tensor-lifetime shapes.

The underlying problem -- assign each tensor's residency episode a
non-negative address such that no two time-overlapping episodes share
address space, keeping the peak used address within budget -- is the
classic "offline dynamic storage allocation" problem: NP-hard in general,
but well studied. This uses Best-Fit-Decreasing (sort episodes by size,
largest first; place each at the smallest sufficient gap among its
time-overlapping already-placed neighbors) -- a standard, well-understood
2-approximation for this exact problem (all lifetimes known in advance,
which is true here: resident_action is fully decided before placement
ever runs). decide_pinning()'s own capacity check already guarantees the
*sum* of sizes of any set of time-overlapping pinned tensors never
exceeds the budget (a valid lower bound on the true minimum addressable
span needed); this placement strategy is chosen to get as close to
achieving that bound as a fast heuristic reasonably can, rather than
inheriting a strategy shaped around a different source system's own
typical allocation patterns.
"""
from typing import Dict, List, Tuple


class OnsramPlacementError(RuntimeError):
    """
    Raised when place_tensors() cannot fit a resident_action's episodes
    into memory_budget_bytes -- a real capacity/fragmentation failure of
    this placement heuristic (not necessarily a bug: general dynamic
    storage allocation with variable-size objects is NP-hard, so a fast
    heuristic can legitimately fail even when decide_pinning()'s own
    aggregate byte-sum check says the combination should fit). This is a
    genuine, meaningful signal about OnSRAM's own two-phase
    (decide-then-place) architecture on a given model/budget, not
    something to silently swallow.
    """
    def __init__(self, message: str, *, tensor_id: int = None, reason: str = None):
        super().__init__(message)
        self.tensor_id = tensor_id
        self.reason = reason


def _extract_episodes(resident_action: Dict[Tuple[int, int], str]
                       ) -> List[Tuple[int, int, int]]:
    """
    Maximal contiguous (by consecutive integer t) runs of 'C'/'P'
    residency per tensor -- an episode needs one stable address for its
    whole span. OnSRAM's own resident_action only ever contains 'C'/'P'
    (see pinning.py's build_resident_action() -- no 'S'/'R', whole-lifetime
    pinning is all-or-nothing), so no filtering by action type is needed
    beyond what's already guaranteed by that function's own contract.
    Returns a list of (tensor_id, t_start, t_end), t_end inclusive.
    """
    ts_by_tensor: Dict[int, List[int]] = {}
    for (a, t) in resident_action:
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


def _best_fit_gap(conflicting_sorted_by_addr: List[Tuple[int, int]], size: int) -> int:
    """
    Among every gap between/before/after already-placed, time-overlapping
    neighbors, return the address of the smallest gap that's still big
    enough (best-fit), or the running high-water mark (cursor) if none
    fits. `conflicting_sorted_by_addr` must already be sorted by address.
    """
    best_offset = None
    best_fit = None
    cursor = 0
    for addr, sz in conflicting_sorted_by_addr:
        gap = addr - cursor
        if gap >= size and (best_fit is None or gap < best_fit):
            best_offset, best_fit = cursor, gap
            if best_fit == 0:
                break
        cursor = max(cursor, addr + sz)
    return cursor if best_offset is None else best_offset


def place_tensors(tensors: Dict[int, object],
                   resident_action: Dict[Tuple[int, int], str],
                   memory_budget_bytes: int) -> Dict[Tuple[int, int], int]:
    """
    Returns spm_plan: (tensor_id, t) -> address, for exactly the
    (tensor_id, t) pairs resident_action marks 'C' or 'P'.

    Algorithm: extract episodes, sort by size descending (ties: earlier
    start first, then tensor id, for determinism) -- Best-Fit-Decreasing
    -- then place each at the smallest sufficient gap among its
    time-overlapping already-placed neighbors. Raises OnsramPlacementError
    (not a silent wrong answer) if a placement can't fit the budget.
    """
    episodes = _extract_episodes(resident_action)
    if not episodes:
        return {}

    def sort_key(ep):
        a, t_start, t_end = ep
        return (-tensors[a].size_bytes, t_start, a)

    episodes.sort(key=sort_key)

    placed: List[Tuple[int, int, int, int]] = []  # (address, size, t_start, t_end)
    address_by_episode: Dict[Tuple[int, int], int] = {}  # (tensor_id, t_start) -> address

    for (a, t_start, t_end) in episodes:
        size = tensors[a].size_bytes
        conflicting = sorted((addr, sz) for (addr, sz, s, e) in placed
                              if s <= t_end and t_start <= e)
        address = _best_fit_gap(conflicting, size)
        if address + size > memory_budget_bytes:
            raise OnsramPlacementError(
                f"place_tensors(): tensor {a}'s episode [{t_start},{t_end}] needs "
                f"{size} bytes but no gap fits in the {memory_budget_bytes}-byte budget "
                f"against time-overlapping placed episodes {conflicting} -- a real "
                f"placement/fragmentation failure of this Best-Fit-Decreasing heuristic "
                f"(general dynamic storage allocation is NP-hard; see this module's "
                f"docstring), not necessarily a bug in the pinning decision itself, which "
                f"only checked that the *total* bytes of overlapping tensors fit, not "
                f"whether they're contiguously placeable.",
                tensor_id=a, reason='capacity_or_fragmentation')
        placed.append((address, size, t_start, t_end))
        address_by_episode[(a, t_start)] = address

    episode_start_of: Dict[Tuple[int, int], int] = {}
    for (a, t_start, t_end) in episodes:
        for t in range(t_start, t_end + 1):
            episode_start_of[(a, t)] = t_start

    spm_plan: Dict[Tuple[int, int], int] = {}
    for (a, t), action in resident_action.items():
        if action in ('C', 'P'):
            spm_plan[(a, t)] = address_by_episode[(a, episode_start_of[(a, t)])]

    from spm_common.spm_allocator import SpmAllocator  # generic, paper-agnostic checker
    try:
        SpmAllocator(tensors, spm_plan, resident_action, memory_budget_bytes).replay_all()
    except Exception as e:
        raise OnsramPlacementError(
            f"place_tensors(): self-verification failed after placement -- this "
            f"indicates a real bug in this module, not a heuristic limitation: {e}",
            reason='self_verification_failed') from e

    return spm_plan
