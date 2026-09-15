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

## Problem 4: SCALE-Sim's resident-buffer override is silently ignored in USER bandwidth mode

**What happened.** OnSRAM's Phase D mechanism for crediting a pinned tensor's read as a free SPM hit (installing a custom read-buffer class in place of the normal one) works correctly under the default `CALC` bandwidth mode, but silently stops working the moment the config is switched to `InterfaceBandwidth: USER` — needed specifically to set an explicit external-memory bandwidth number (e.g. to match the paper's stated 32 GBps) rather than letting SCALE-Sim derive an implicit one from array width. Under `USER` mode, every pinned tensor's read is still charged as a full DRAM fetch, with no error or warning of any kind — the credit is just silently zero.

**Root cause, confirmed by reading the engine source directly.** `scalesim/memory/double_buffered_scratchpad_mem.py`'s `set_params()` only honors a caller-supplied read-buffer class override inside its `if self.estimate_bandwidth_mode:` branch:
```python
if self.estimate_bandwidth_mode:
    self.ifmap_buf = ifmap_buf_class() if ifmap_buf_class else rdbuf_est()
    ...
else:
    self.ifmap_buf = rdbuf()   # ifmap_buf_class silently discarded here
```
`estimate_bandwidth_mode` is `True` exactly when bandwidth mode is `CALC` and `False` exactly when it's `USER` (confirmed in the calling code, which sets `estimate_bandwidth_mode = False` in the `use_user_dram_bandwidth()` branch). So in `USER` mode, the `else` branch always constructs the plain default buffer class, and the subsequent line that flags a specific tensor as fully resident is setting an attribute on an object that never checks it — a silent no-op, not an error.

This is asymmetric: the equivalent override for the *write* buffer (used to model a freshly-produced output staying on-chip) is assigned directly on the memory-system object *before* `set_params()` runs, and `set_params()` never reassigns that field afterward — so the write-buffer override survives in both bandwidth modes. Only the read-buffer override is broken, and only in `USER` mode.

**Impact.** Any Phase D run using a `USER`-mode config (needed for every bandwidth value that doesn't happen to match what `CALC` mode would have derived on its own) understated OnSRAM's true DRAM savings — the entire benefit of pinning an *input* activation went unmodeled, while the benefit of a freshly-produced *output* staying on-chip was still counted correctly. A run in the default `CALC`-mode config was unaffected.

**Fix.** Patched in `scalesim/memory/double_buffered_scratchpad_mem.py`'s `set_params()`: the ifmap and filter read-buffer classes are now each decided independently (`estimate_bandwidth_mode OR that buffer's own class override`), so an override is honored in both bandwidth modes, and overriding one buffer never changes how the other is modeled. This surfaced a second, related gap: `single_layer_sim.py` calls `set_read_buf_prefetch_matrices()` whenever the config as a whole is in `USER` mode, assuming both buffers are always the bank/port-modeling class that uses a prefetch matrix — no longer true once a `ReadBufferEstimateBw`-based override can appear in that mode too. Fixed by guarding each buffer's prefetch-matrix call with `hasattr(..., 'set_fetch_matrix')` instead of a mode check, so a buffer that doesn't use a prefetch matrix (true for the override, and already true for `CALC` mode's own default) simply has nothing installed, rather than crashing.

**Verified.** Regression-tested against `CALC`-mode results from before the fix (byte-for-byte identical: 84.21% DRAM reduction, 210,864 / 1,337,998 ifmap/ofmap credit bytes on `resnet20_cifar10`@1MB) and confirmed the fix itself under `USER` mode on the same model+budget: ifmap credit went from a silent `0` to a real `3,012,736` bytes, changing the measured speedup from a misleading ~1.003x to a genuine 1.87x once the paper's actual 39×39-array/32GBps ratio makes most baseline layers memory-bound.

**Status.** Fixed.

---

## Problem 5: A config's bandwidth value must divide evenly across its bank count, with no clear error until it doesn't

**What happened.** Setting `Bandwidth: 32` in `USER` mode, while leaving the inherited `IfmapSRAMBankNum`/`FilterSRAMBankNum` at `10`, fails with `AssertionError: overall bandwidth must be divisible by total number of banks, number of banks = 10, bandwidth of each as 3, total bandwidth = 32`.

**Root cause.** `scalesim/memory/read_buffer.py` computes `bw_per_bank = req_gen_bandwidth // num_bank` (integer floor division) and then asserts `bw_per_bank * num_bank == req_gen_bandwidth`. Any bandwidth value not evenly divisible by the configured bank count trips this — a real, intentional invariant (bandwidth genuinely can't split evenly across banks otherwise), but one with no mention in the config format's own documentation, so it only surfaces once a specific numeric combination happens to violate it. The stock `scale.cfg` (`Bandwidth: 10`, `*SRAMBankNum: 10`) never hit this only because both numbers happened to already match.

**Fix.** Set `IfmapSRAMBankNum`/`FilterSRAMBankNum` to `1` in any config that needs a specific, otherwise-arbitrary bandwidth value — with a single bank, the divisibility check is trivially satisfied for any bandwidth number.

---

## Summary for write-up

- The reference implementation contains a genuine off-by-one that inflates its own reported pinning success rate beyond what's physically achievable; this port uses the paper's own stated (inclusive) definition instead, at the cost of a lower — but real — pin ratio.
- Making the Overwrite Optimization work correctly under real byte-addressed constraints requires one explicit exception to the general live-range rule, specific to reclaim-source tensors.
- The paper's own feasibility check (aggregate byte-count only) is provably insufficient to guarantee a real placement exists, once actual SPM addresses are required instead of a closed-form bandwidth estimate — confirmed empirically on a large branched network, where the greedy algorithm's natural tendency to fill the budget to near-100% utilization leaves too little slack for any offline placement heuristic to reliably succeed.
- Beyond the OnSRAM algorithm itself, wiring into real SCALE-Sim surfaced two genuine engine-level bugs: a bandwidth/bank-count mismatch fails with a correct but undocumented assertion; and — far more significantly — the engine's own mechanism for crediting a resident read as free was silently inert in `USER` bandwidth mode, understating measured DRAM savings for any experiment that needs an explicit, paper-matching bandwidth value rather than the engine's own derived default. The second one is now fixed and regression-tested; on the same model and budget, fixing it changed the measured speedup from a misleading ~1.003x to a genuine 1.87x once the paper's real hardware ratio (39×39 array, 32 GBps) is actually in effect.
