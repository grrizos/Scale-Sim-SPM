# SCALE-Sim Optimization Survey — Round 2 Candidates

Status: **survey only** — profiled and read against the current working tree (Stages 1-4 from
`PERFORMANCE_FIXES.md` and the trace-accumulation/crash fixes from `SMM_PERFORMANCE.md` are all
confirmed present in the current source). Nothing in this document has been implemented yet.

Method: `cProfile` on the same setup as the original diagnosis — first 3 layers of
`topologies/conv_nets/mobilenet.csv`, default `configs/scale.cfg` (`ws` dataflow, `InterfaceBandwidth:
CALC`, 32×32 array, 64KB/operand) — run fresh against today's code, then read the flagged functions'
current source to confirm each finding before writing it down here.

## Baseline for this round

**21.944s, 27,486,190 function calls** for the same 3-layer profile that measured 48.7s / 36.9M calls
before Stages 1-4. That's directionally consistent with the ~42-50% wall-clock gains already reported
(cProfile overhead means this isn't a like-for-like wall-clock comparison, but the drop is in the
right ballpark).

## Findings

| # | Location | Problem | Measured cost (this profile) | Ease | Status |
|---|---|---|---|---|---|
| A | `read_buffer_estimate_bw.py`, `read_buffer.py`, `write_buffer.py`, `double_buffered_scratchpad_mem.py` — all 6 `np.savetxt(...)` call sites | `np.savetxt` formats and writes row-by-row in pure Python; called 18× (6 per layer: ifmap/filter/ofmap × SRAM-side + DRAM-side traces) | **3.675s tottime — ~17% of total profiled time.** The single largest tottime consumer in the entire run, ahead of every per-cycle function | Easy–Medium | New finding |
| B | `double_buffered_scratchpad_mem.py`: `get_ifmap_sram_start_stop_cycles`, `get_filter_sram_start_stop_cycles`, `get_ofmap_sram_start_stop_cycles` | Nested pure-Python double loop scanning the trace matrix row-by-row, then column-by-column within each row, to find the first/last row containing a real (non -1) request | Filter version alone: 167ms / 3 calls (~56ms/layer) in this profile. Filter traffic is sparse (few real entries per row), so the scan runs long before it can break early — likely worse on layers/models with even sparser filter reuse | Easy | New finding |
| C | `write_buffer.py`: `store_to_trace_mat_cache` | Called once **per scalar element** (1.3M calls across 3 layers) doing trivial work — the cost is pure CPython function-call overhead, not algorithmic complexity | 1.26s tottime (~5.7% of total) | Medium–Hard | New finding |
| D | `read_buffer_estimate_bw.py`: `manage_prefetches`, `check_hit`, `service_reads`; `write_buffer.py`: `service_writes` | Already O(1) per operation (dict-based, from Stages 3-4). Remaining cost is CPython per-call overhead × millions of calls (4.6M / 4.6M / 295K / 148K respectively for this profile) | ~2.9-3.0s tottime each for the top two; together with `check_hit`/`service_writes`, roughly **half** the total profiled time | Hard | Confirms the existing roadmap's "core loop" flag — no more algorithmic slack here |
| E | `double_buffered_scratchpad_mem.py:264`, the `tqdm`-wrapped per-cycle loop inside `service_memory_requests` | A single `tqdm` object wraps the *entire* per-layer cycle loop (constructed once per layer, not once per cycle like the original Fix #1 bug) | **~0.09s across the whole run (~0.4%).** Confirmed via `tqdm/std.py:__iter__`'s profiled cost | Trivial | Checked, low priority — a different, much smaller shape than the original bug |
| F | `systolic_compute_os.py`: 3× `tqdm(total=..., disable=True)` (lines 261, 313, 363) | Same shape as E, in the OS-dataflow file | Not measured — this profile uses `ws` dataflow only | Trivial | Flagged by inspection only, unverified |
| G | `operand_matrix.py`: `calc_ifmap_elem_addr` | Already vectorized (meshgrid + `np.divmod`) | 0.5s tottime / 3 calls — scales with matrix size, this is real work, not overhead | — | Ruled out, noted so it isn't re-investigated |

## Priority read

1. **A (`np.savetxt`) is the clear first move.** Biggest single measured cost in this profile
   (~17%), touches only output serialization (not simulated values), and is the easiest of
   everything here to verify safe — diff the written files before/after, same as every fix in
   `PERFORMANCE_FIXES.md`. Candidate direction: swap to a faster CSV writer for large numeric arrays
   (e.g. `pandas.to_csv`, or a hand-rolled vectorized formatter) — needs its own before/after timing
   once a specific replacement is picked, not assumed here.
2. **B (start/stop-cycle scans)** — small in this 3-layer profile but essentially free to fix: the
   exact same "replace a hand-rolled Python loop with a vectorized numpy check" pattern already
   proven safe in the original round's Fix #2. Natural to bundle with A.
3. **C (`store_to_trace_mat_cache` batching)** — real, measurable cost, but the fix means changing
   *how often* the function is called (batch a whole request line in one call instead of one call per
   element), not just what happens inside it — more design work and its own verification pass, same
   caution level as the original Stage 3/4 fixes.
4. **D is the "big rewrite" the acceleration roadmap already anticipated.** This profile confirms
   there's no more low-risk, isolated-function slack left in the per-cycle read/write path — every
   remaining microsecond there is CPython call overhead, not a bad algorithm. Closing more of the gap
   toward 70%+ likely has to go through here next: batching multiple cycles per Python-level call,
   JIT/Cython-compiling the innermost loop, or parallelizing independent layers — not another
   isolated-function patch.
5. **E/F** — low priority. E is confirmed small; F is unverified and dataflow-specific (only matters
   once OS-dataflow runs are actually profiled). Fix opportunistically if touching those files anyway,
   not worth a dedicated pass.

## What this changes about the 70%+ target

Stages 1-4 drove the per-cycle hot path down to its CPython call-overhead floor (item D). The two
items with real, *unexploited* headroom — A and B — sit **outside** the per-cycle loop (A is
post-layer file I/O, B is post-layer reporting), so fixing them is additive to the existing 42-50%,
not a re-optimization of already-fixed code. Rough arithmetic from this profile: A alone is ~17% of
this run's profiled time. That is not the same number as "17% additional wall-clock speedup on a full
run" — same caveat as the original plan's "Stages 1-3 target ~70% of profiled time" line, which also
didn't translate 1:1 into wall-clock percent. Treat as a promising, worth-trying next step, to be
re-measured end-to-end (full 27-layer MobileNet, real wall clock, golden-trace diff) once implemented
— not a promised number.

## What wasn't re-verified in this pass

- No full 27-layer wall-clock run was profiled here (only the same 3-layer subset used for the
  original diagnosis) — the relative sizes above should hold reasonably well since both the per-cycle
  loop and the trace-writing cost scale with total cycles, but that's an inference, not a
  measurement.
- `os`/`is` dataflow paths (item F) weren't profiled at all — this run only exercises `ws`.
- Sparsity-compression loops and Ramulator per-request loops remain out of scope, same as the
  original plan (`operand_matrix.py` sparsity paths and `compression.py`, gated behind
  `SparsitySupport`/`UseRamulatorTrace`, both default off).
