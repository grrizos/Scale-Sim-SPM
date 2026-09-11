# cosma/helpers/belady_policy.py
"""
The paper's Belady's-algorithm tensor-replacement baseline (§V-A2):
"Belady's algorithm, which always replaces the tensor to be accessed in
the furthest future." Classic offline furthest-in-future/MIN cache
eviction, applied per-tensor with full lookahead over the whole remaining
fixed schedule -- legitimate here (not an artificially crippled version)
since Belady/MIN is *defined* as the offline-oracle algorithm, and both
schedule variants (schedule_variants.py) are already fully fixed before
replacement ever runs -- matching the paper's own framing: a fair baseline
against COSMA's own ILP, which also sees the whole graph, not a
window-limited one.

Implements the choose_victims_fn contract shared with
ilp_greedy_policy.py -- see replacement_engine.simulate_replacement()'s
docstring. Purely size-blind: ranks only by timing, never by how many
bytes an eviction actually frees -- this is precisely what the paper's own
empirical finding says makes it non-optimal relative to the greedy-ILP
policy (see ilp_greedy_policy.py's docstring for the mechanical reason
why, and its __main__ for a verified concrete divergence).
"""
from typing import Dict, List


def choose_victims(candidates: List[int], deficit_bytes: int, current_t: int,
                    tensors: Dict[int, object],
                    consumer_positions: Dict[int, List[int]]) -> List[int]:
    """
    Ranks candidates by furthest next-access position (a tensor never
    accessed again ranks first -- infinite distance), ties broken by
    tensor id for determinism, and takes greedily from the front until
    deficit_bytes is covered.

    "Rank once, take from the front" is behaviorally identical to "evict
    the single furthest-future one, then re-rank, repeat, until covered":
    a candidate's next-access distance is a static graph fact (doesn't
    change as OTHER candidates get evicted), so re-ranking after each
    eviction within the same decision point would never change the order
    -- this version just avoids re-sorting N times for the same result.
    """
    def next_access(a):
        future = [p for p in consumer_positions[a] if p >= current_t]
        return min(future) if future else float('inf')

    ranked = sorted(candidates, key=lambda a: (-next_access(a), a))
    victims: List[int] = []
    freed = 0
    for a in ranked:
        if freed >= deficit_bytes:
            break
        victims.append(a)
        freed += tensors[a].size_bytes
    return victims


if __name__ == '__main__':
    class _T:
        def __init__(self, size_bytes):
            self.size_bytes = size_bytes

    # Never-accessed-again candidate must rank first (infinite distance),
    # ahead of one with a large but finite distance.
    tensors = {1: _T(50), 2: _T(50), 3: _T(50)}
    consumer_positions = {1: [], 2: [1000], 3: [5]}
    victims = choose_victims([1, 2, 3], 40, 0, tensors, consumer_positions)
    assert victims == [1], f"expected tensor 1 (never reused) evicted first, got {victims}"
    print(f"Never-reused-first case: PASS, evicted {victims}")

    # Multi-victim coverage, furthest-first order.
    victims2 = choose_victims([1, 2, 3], 90, 0, tensors, consumer_positions)
    assert victims2 == [1, 2], f"expected [1 (inf), 2 (dist 1000)] in that order, got {victims2}"
    print(f"Multi-victim, furthest-first case: PASS, evicted {victims2} in order")

    print("belady_policy.py: PASS")
