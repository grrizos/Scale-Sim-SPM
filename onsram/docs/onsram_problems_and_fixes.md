# OnSRAM Port — Problems Found and How They Were Handled

This documents real issues discovered while porting OnSRAM-Static's algorithm (FoM scoring, greedy whole-interval pinning with the Overwrite Optimization, BFS-DFS scheduling) into a real, byte-addressed, cycle-accurate simulation pipeline. Each one surfaced only once the algorithm was driven by an actual physical-consistency checker and a real simulator — none of them are visible from the paper's own description or its own bandwidth-based evaluation methodology (§6), which never modeled real SPM addresses at all.

---

## Problem 1: The reference implementation's live-range calculation undercounts a tensor's true footprint

**What happened.** The original OnSRAM-Static prototype (`get_live_timesteps()`) computes a tensor's SPM-occupancy window as `range(produced_at_ts, last_used_at_ts)` — an **exclusive** end, dropping the timestep the tensor is actually last read. This directly affects the pinning decision's capacity check.

**Why it's a bug, not a design choice.** The paper itself defines pinning (§4.2) as fitting in SPM "for all the timesteps that it is alive" — an inclusive definition. Verified directly against a real byte-addressed allocator: two tensors of 1.53MB each, produced at consecutive timesteps in a producer→consumer chain, both got accepted for pinning under the reference's exclusive-range accounting (each tensor's counted usage window never overlaps the other's). But physically, the consuming node needs its input (still being read) and its output (just being written) resident **simultaneously** — 3.06MB against a 2MB budget, a real capacity violation. The reference's own reported speedup numbers for this case are not physically achievable; the bug is *load-bearing* for those numbers, not incidental.

**Fix.** Use the inclusive range everywhere by default. This drops the achievable pin ratio on the same test case from the reference's reported 29–30/31 to a physically valid 27/30 — lower, but actually realizable (independently re-verified via a live byte-addressed allocator replay).

---

## Problem 2: Fixing Problem 1 breaks the Overwrite Optimization entirely

**What happened.** The Overwrite Optimization (§4.2's "overwrite candidate" check) is supposed to let a dying tensor's SPM space be reused by a new tensor born at the same instant from the same producing/consuming operation — e.g. an in-place-style handoff. Once Problem 1's fix (inclusive range, applied uniformly) was in place, the Overwrite Optimization stopped ever functioning: any handoff between comparably-sized tensors now always double-booked and failed, exactly reproducing Problem 1's own bug in a new form.

**Root cause.** The Overwrite Optimization's entire value proposition is that two tensors can share one physical address across the instant one dies and the other is born. A real byte-addressed allocator has no representation for two different tensor IDs occupying the same address at the same timestep — the only way to express the handoff is for the dying tensor to vacate its space *before* the new tensor claims it, i.e. exactly the exclusive-range behavior — but only for tensors genuinely involved in a reclaim, never as a blanket rule.

**Fix.** A two-part rule: inclusive range by default (Problem 1's fix stays in place for every tensor), with one explicit, narrow exception — a tensor that the Overwrite Optimization actually used as a reclaim source for another tensor's fit vacates one timestep early. This restores the Overwrite Optimization as a real, working mechanism. Disclosed cost: that one tensor's very last real read gets modeled as a DRAM fetch instead of an SPM hit once wired into real simulation, since true address-aliasing isn't representable — a conservative simplification, not a correctness bug.

---

## Problem 3: The paper's own capacity check doesn't guarantee a physically realizable placement

**What happened.** On a large, heavily-branched network (InceptionResNetV2, 335 layers), the pinning decision reports success (e.g. 322/335 tensors pinned at a 3MB budget), but the subsequent step of assigning each pinned tensor a concrete SPM byte address fails — a specific tensor's storage episode needs more space than any available gap in the address layout, even though the total budget hasn't nominally been exceeded.

**Root cause, confirmed empirically, not assumed.** The pinning decision's own feasibility check (`CheckSPM(size, StartTS, EndTS)` in the paper's own Figure 4) is a purely *aggregate* check: it sums the sizes of every tensor simultaneously alive and compares that single number against the budget. It implicitly assumes perfect packing — that any combination of tensors whose total bytes fit under budget can always be laid out in actual contiguous address space. That assumption is false in general: assigning fixed-size objects with known lifetimes to non-overlapping address ranges (given only an aggregate bound) is the classic "dynamic storage allocation" problem, which is NP-hard — a fast placement heuristic can legitimately fail even when the aggregate check passes.

Measured directly on the failing cases: the true aggregate peak simultaneous demand across the whole schedule came out to **98.6%** and **99.7%** of the budget respectively (2MB and 3MB test cases) — only 1.4% and 0.31% of headroom. This is not a coincidence specific to one budget: the pinning algorithm greedily accepts tensors by Figure-of-Merit score *until the budget is full*, which by construction tends to land very close to 100% aggregate utilization regardless of the specific budget value. At that margin, essentially any placement heuristic — not just one specific implementation choice — is likely to fail, because there is almost no slack left to absorb any fragmentation at all. Confirmed by testing three structurally different placement orderings (size-descending, start-time-first, a combination of both) against the identical pinning decision: all three failed, each at a different tensor, since none has enough look-ahead to know that the tightest point in the whole schedule occurs many timesteps after the point where they run out of good options.

**Why the paper doesn't address this.** Checked directly: the paper's own performance evaluation (§6, "Performance Model") uses a closed-form, bandwidth-centric latency estimate — `max(compute_time, data_transfer_time)` — driven only by a boolean *pinned/not-pinned* flag per tensor. It never simulates or requires a real byte-addressed memory at all, so the placement/fragmentation problem this port hit simply cannot arise in the paper's own methodology. This gap is a consequence of deliberately going further than the paper did — wiring the algorithm into a real, byte-addressed simulator to get real cycle-accurate numbers — not a mistake in porting the paper's algorithm faithfully.

**Status.** Left as an honest, disclosed failure rather than engineered around. The pinning decision and the placement step both do exactly what the paper specifies; the mismatch between "aggregate bytes fit" and "the specific combination is contiguously placeable" is a genuine, real limitation of extending a bandwidth-only algorithm into a physically realized one, worth stating plainly rather than papering over with a more complex placement algorithm that would still not carry any guarantee in the general case.

---

## Summary for write-up

- The reference implementation contains a genuine off-by-one that inflates its own reported pinning success rate beyond what's physically achievable; this port uses the paper's own stated (inclusive) definition instead, at the cost of a lower — but real — pin ratio.
- Making the Overwrite Optimization work correctly under real byte-addressed constraints requires one explicit exception to the general live-range rule, specific to reclaim-source tensors.
- The paper's own feasibility check (aggregate byte-count only) is provably insufficient to guarantee a real placement exists, once actual SPM addresses are required instead of a closed-form bandwidth estimate — confirmed empirically on a large branched network, where the greedy algorithm's natural tendency to fill the budget to near-100% utilization leaves too little slack for any offline placement heuristic to reliably succeed.
