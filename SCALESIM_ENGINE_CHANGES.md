# SCALE-Sim Engine Execution Changes — Summary

Scope: only changes under `scalesim/` that affect how the simulator itself runs (speed or
correctness of the engine). Excludes everything in `cosma/` and `onsram/` (those are algorithm
ports built *on top of* the engine, documented separately in their own `docs/` folders). Distilled
from `PERFORMANCE_FIXES.md`, `SIMPLE_PERFORMANCE_EXPLANATION.md`, `SMM_PERFORMANCE.md`,
`OPTIMIZATION_SURVEY.md`, `onsram/docs/onsram_problems_and_fixes.md`, and `cosma/docs/STATUS.md` /
`ITERATION_HISTORY.md`.

## Speed fixes (implemented, verified byte-identical output)

| # | File(s) | What | Gain |
|---|---|---|---|
| 1 | `write_buffer.py`, `read_buffer.py` | Removed per-cycle `tqdm` object construction (bar was always `disable=True`, never rendered) | 9% |
| 2 | `systolic_compute_{ws,os,is}.py` | Vectorized the diagonal-flatten prefetch-matrix build (was one element at a time) | ~5% |
| 3 | `read_buffer_estimate_bw.py` | `check_hit()`: linear scan → dict lookup (`addr -> most_recently_finalized_set_id`) | **30%**, biggest single win |
| 4 | `read_buffer.py` | `active_buffer_hit()`: linear scan → dict lookup; `set_fetch_matrix()` vectorized | ~8% |

Combined (#1-4): full 27-layer MobileNet 4m15s → 2m28s (~42%); 7-layer USER-mode subset 2m0s →
59.6s (~50%). Verified: standalone unit tests old-vs-new logic on synthetic inputs, golden-trace
regression suite (5/5 scripts, zero diff), full-model before/after output byte-identical.

| 5 | `write_buffer.py`, `read_buffer_estimate_bw.py`, `read_buffer.py` | Trace accumulation was repeated `np.concatenate` (copies the whole trace so far on every drain/prefetch event → O(N²)). Now accumulates chunks in a list and concatenates once at read time (`write_buffer.py` instead over-allocates and doubles on overflow, since its trace is read mid-simulation) | Only shows up with very small buffers (e.g. SMM policies); one measured case went 26.2s → 13.9s (~2x) |

## Correctness / crash fixes (implemented, verified)

| # | File | Bug | Fix |
|---|---|---|---|
| 6 | `read_buffer_estimate_bw.py` | `num_items_per_set = floor(total_size_elems / 100)` hits 0 for buffers < 100 elements → chunks never finalize → crash in `complete_all_prefetches()` | Floor of 1: `max(1, floor(...))` |
| 7 | `write_buffer.py` | `drain_buf_size = total - active_buf_size` can hit 0 for a 1-element buffer → buffer can never drain → deadlock/crash | Floor of 1: `max(1, total - active_buf_size)` |
| 8 | `double_buffered_scratchpad_mem.py` | Caller-supplied read-buffer class override (`ifmap_buf_class`/`filter_buf_class`) was only honored when `estimate_bandwidth_mode` was True (`CALC` mode); silently ignored in `USER` mode, so a resident tensor's read was still charged as a full DRAM fetch with no warning | Each buffer now takes the override path independently: `estimate_bandwidth_mode OR that buffer's own class override`. Also guarded `set_fetch_matrix()`/`complete_all_prefetches()` calls with `hasattr(...)` instead of a mode check, since an override doesn't implement those | **Currently uncommitted** (working-tree change only) |

Both #6/#7 only change behavior for buffers small enough to hit the edge case — no-op for
normal-sized (default 64KB) buffers. #8 is the fix behind the OnSRAM "USER-mode bandwidth" work
discussed earlier in this project.

## New engine extension point (additive, no-op unless used)

| File | What |
|---|---|
| `cosma_resident_buffers.py` (new) | `CosmaResidentReadBuffer`/`CosmaResidentWriteBuffer` — subclasses of the real read/write buffer classes that override only "this tensor is already resident, skip the fetch/drain"; every other path delegates unchanged to stock SCALE-Sim logic. Installed via #8's `ifmap_buf_class`/`filter_buf_class` params. Regression-verified byte-identical to stock when the resident flag is left off. |

## Known gotcha — config-only, no code fix

`read_buffer.py` asserts `bw_per_bank * num_bank == req_gen_bandwidth` (integer floor division) —
a `USER`-mode bandwidth value that doesn't divide evenly across the configured SRAM bank count
throws an undocumented `AssertionError`. No engine change; the workaround is setting
`IfmapSRAMBankNum`/`FilterSRAMBankNum` to `1` in any config needing a specific bandwidth value.

## Surveyed, not yet implemented

`OPTIMIZATION_SURVEY.md` profiled further candidates on top of fixes #1-4: `np.savetxt` trace
writing (~17% of profiled time, largest single remaining cost), the start/stop-cycle scan loops in
`double_buffered_scratchpad_mem.py` (~small currently, could grow on sparser models), and
per-element calls in `write_buffer.py`'s `store_to_trace_mat_cache` (~5.7%). None of these have
been implemented yet.
