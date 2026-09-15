# onsram/onsram_helpers/pinning.py
"""
Greedy whole-interval-only SPM pinning + Overwrite Optimization, ported
from /home/george/trim/spm_management/onsram_weights.py's
decide_spm_pinning(), plus construction of the resident_action/spm_plan
dict shapes SpmAllocator (spm_common/spm_allocator.py, imported directly
as a generic physical-consistency checker) expects. Address assignment itself
(build_spm_plan) uses OnSRAM's own placement algorithm -- see
onsram_helpers/placement.py's module docstring for why.

Deliberate deviation from the reference's own get_live_timesteps():
the reference's general-case branch returns an EXCLUSIVE-end range
(range(produced_at_ts, last_used_at_ts), dropping the last-use
timestep) for any tensor where produced != last_used. Empirically
verified this is only correct for tensors the Overwrite Optimization
actually reclaims (see below) -- for every other tensor it's a real
undercounting bug relative to the OnSRAM paper, which defines pinning
(Sec 4.2) as fitting in SPM for a tensor's "entire lifetime." This port
therefore uses the INCLUSIVE range (live_range()) as the default
everywhere, with one explicit, narrow exception:

**Overwrite Optimization's address hand-off.** SpmAllocator has no
concept of two different tensor ids sharing one address at the same
timestep (spm_common/spm_allocator.py's _allocate() rejects any
address overlap outright), and this port's own placement algorithm
(onsram_helpers/placement.py) has no in-place/aliasing buffer sharing
either. So the only way to express "tensor A's buffer gets reused by
tensor B, born from the very node that consumes A, at A's last-use
timestep" is for A to vacate (no resident_action entry) at that shared
timestep, freeing generic capacity one step before B's own 'C' claims
space there -- i.e. exactly the reference's exclusive-range treatment,
but ONLY for tensors decide_pinning() actually used as a reclaim source
for some other pinned tensor's hand-off (tracked via
reclaimed_source_ids below), never as a blanket rule. Confirmed
necessary empirically: on
MobileNet, tensor_3 (produced t=0, last used t=1, 1.53MB) and tensor_6
(produced t=1, last used t=2, 1.53MB) both get pinned via the Overwrite
Optimization's reclaim check at their shared t=1 hand-off -- if tensor_3
stayed resident through t=1 (the naive inclusive rule), it would
collide with tensor_6's own 'C' at t=1 (3.06MB against a 2MB budget);
letting tensor_3 vacate at t=1 instead (this exception) makes the plan
physically realizable, matching what the Overwrite Optimization's own
accounting in decide_pinning() already assumed when it let tensor_6 fit.
The disclosed cost: tensor_3's very last real read (by the node that
also produces tensor_6) gets modeled as a DRAM fetch instead of an SPM
hit in Phase D, since this port's placement layer can't express true
address aliasing -- a conservative simplification, not a correctness bug.
"""
from typing import Dict, Set, Tuple

from .placement import place_tensors


def live_range(produced_at_ts: int, last_used_at_ts: int) -> range:
    """
    Inclusive live range: range(produced_at_ts, last_used_at_ts + 1). A
    tensor occupies SPM from the timestep it's produced through the
    timestep its last consumer actually reads it. This is the default
    used everywhere except tensors flagged as Overwrite Optimization
    reclaim sources -- see module docstring.
    """
    if produced_at_ts < 0:
        return range(0)
    return range(produced_at_ts, last_used_at_ts + 1)


def decide_pinning(tensors: Dict[int, object],
                    produced_at_ts: Dict[int, int],
                    last_used_at_ts: Dict[int, int],
                    tensor_fom: Dict[int, float],
                    schedule_len: int,
                    capacity_bytes: int) -> Tuple[Dict[int, bool], Set[int]]:
    """
    Direct id-keyed port of decide_spm_pinning(): global FoM-descending
    greedy selection (tie-break: earlier production, then smaller size),
    whole-interval-only fit check, with the Overwrite Optimization's
    reclaimable-space check preserved (an already-pinned tensor `other`
    dying exactly at `tensor`'s birth timestep, consumed by the same
    node that produces `tensor`, lets `tensor` reuse that freed space at
    the hand-off timestep only).

    Returns (pinned, reclaimed_source_ids): `pinned` is {tensor_id: bool}
    for every tensor_id in `tensors`; `reclaimed_source_ids` is the set
    of tensor ids that contributed reclaimable space toward some other
    pinned tensor's hand-off fit -- these must vacate one timestep early
    when build_resident_action() materializes them (see module
    docstring). A source's last_used_at_ts always exactly equals the
    reclaiming tensor's produced_at_ts by construction of the check
    below, so no separate timestep bookkeeping is needed here.
    """
    pinned: Dict[int, bool] = {tid: False for tid in tensors}
    reclaimed_source_ids: Set[int] = set()
    usage_by_ts = [0 for _ in range(schedule_len)]

    candidates = sorted(
        tensors.keys(),
        key=lambda tid: (-tensor_fom[tid], produced_at_ts[tid], tensors[tid].size_bytes),
    )

    for tid in candidates:
        produced = produced_at_ts[tid]
        last_used = last_used_at_ts[tid]
        live_ts = [ts for ts in live_range(produced, last_used) if 0 <= ts < schedule_len]
        if not live_ts:
            continue

        size = tensors[tid].size_bytes
        if size > capacity_bytes:
            continue

        # --- Overwrite Optimization: reclaimable space at the hand-off timestep ---
        reclaimable = 0
        start_ts = produced
        reclaim_sources = []
        for other_id, other in tensors.items():
            if pinned[other_id] and last_used_at_ts[other_id] == start_ts:
                if tensors[tid].producer_layer in other.consumer_layers:
                    reclaimable += other.size_bytes
                    reclaim_sources.append(other_id)
        effective_start_size = max(0, size - reclaimable)

        fits_everywhere = True
        for ts in live_ts:
            size_to_add = effective_start_size if ts == start_ts else size
            if usage_by_ts[ts] + size_to_add > capacity_bytes:
                fits_everywhere = False
                break

        if fits_everywhere:
            pinned[tid] = True
            reclaimed_source_ids.update(reclaim_sources)
            for ts in live_ts:
                size_to_add = effective_start_size if ts == start_ts else size
                usage_by_ts[ts] += size_to_add

    return pinned, reclaimed_source_ids


def build_resident_action(tensors: Dict[int, object], pinned: Dict[int, bool],
                           produced_at_ts: Dict[int, int],
                           last_used_at_ts: Dict[int, int],
                           reclaimed_source_ids: Set[int]
                           ) -> Dict[Tuple[int, int], str]:
    """
    Translates the whole-lifetime pinned/not-pinned decision into the
    shared per-timestep resident_action vocabulary (confirmed against
    spm_common/spm_allocator.py's step() semantics directly): 'C' at
    the tensor's produced timestep, then 'P' at every subsequent
    timestep through last_used_at_ts INCLUSIVE (live_range) -- EXCEPT for
    tensors in `reclaimed_source_ids`, which vacate one timestep early
    (no entry at last_used_at_ts) so the tensor that reclaimed their
    space can take over generic capacity at that shared timestep -- see
    module docstring for why COSMA's model requires this exception
    rather than true address aliasing. Not-pinned tensors get zero
    entries -- never 'S' (spill only applies to a tensor that was
    resident and got evicted, which never happens under OnSRAM's
    all-or-nothing rule) and never 'R' (retrieve only applies after a
    spill, structurally impossible here).
    """
    resident_action: Dict[Tuple[int, int], str] = {}
    for tid, is_pinned in pinned.items():
        if not is_pinned:
            continue
        produced = produced_at_ts[tid]
        last_used = last_used_at_ts[tid]
        live_ts = list(live_range(produced, last_used))
        if tid in reclaimed_source_ids and len(live_ts) > 1:
            live_ts = live_ts[:-1]
        if not live_ts:
            continue
        resident_action[(tid, live_ts[0])] = 'C'
        for ts in live_ts[1:]:
            resident_action[(tid, ts)] = 'P'
    return resident_action


def build_spm_plan(tensors: Dict[int, object],
                    resident_action: Dict[Tuple[int, int], str],
                    memory_budget_bytes: int) -> Dict[Tuple[int, int], int]:
    """
    Thin wrapper around OnSRAM's own placement algorithm
    (onsram_helpers/placement.py's place_tensors() -- Best-Fit-Decreasing,
    chosen for OnSRAM's own decision patterns rather than inherited from
    a different system's allocator). Raises OnsramPlacementError up to
    the caller on genuine fragmentation -- a real, meaningful signal about
    this model/budget combination, not something to swallow silently.
    """
    return place_tensors(tensors, resident_action, memory_budget_bytes)
