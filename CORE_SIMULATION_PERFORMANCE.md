# SCALE-Sim Core Simulation Performance (Non-Logging)

Scope: only the functions that do genuine per-cycle/per-request **simulation logic** — cache
hit/miss decisions, buffer state, address computation, fetch scheduling. Excludes everything that
exists purely to write trace/report files (`savetxt`, `store_to_trace_mat_cache`,
`TextIOWrapper.write`/`close`, `get_*_sram_start_stop_cycles`) — that's a separate, already
well-understood ~25-35% of runtime (see `PERFORMANCE_FIXES.md`/`OPTIMIZATION_SURVEY.md`) and
trimming I/O further isn't the interesting problem here.

## Where the core-logic time actually goes

| Function | File | % of total wall time | What it does |
|---|---|---|---|
| `service_memory_requests` (own time) | `scalesim/memory/double_buffered_scratchpad_mem.py` | 11-20% | The per-cycle orchestration loop — one Python iteration per simulated clock cycle |
| `service_reads` | `scalesim/memory/read_buffer_estimate_bw.py` | 12-18% | For every requested address, every cycle: hit/miss decision + read-buffer state update |
| `manage_prefetches` | `scalesim/memory/read_buffer_estimate_bw.py` | 11-14% | Groups addresses into sets, decides when a DRAM prefetch fires |
| `check_hit` | `scalesim/memory/read_buffer_estimate_bw.py` | 9-10% | The actual cache-hit lookup (already O(1) dict-based since `PERFORMANCE_FIXES.md`) |
| `service_writes` (own time) | `scalesim/memory/write_buffer.py` | 5-6% | Write-buffer occupancy/drain-timing state machine |
| `calc_ifmap_elem_addr` | `scalesim/compute/operand_matrix.py` | 3-4% | Ifmap demand-matrix address computation (setup phase, not per-cycle) |
| `create_ifmap_prefetch_mat` | `scalesim/compute/systolic_compute_ws.py` | 2-3% | Builds the DRAM fetch-order matrix |
| `dict.get`/`min`/`max`/`set.add` | builtins | ~2-3% each | Generic Python call overhead *of* the functions above |

**Total: roughly 45-55% of wall-clock time is genuine simulation logic**, the rest is
logging/trace I/O (excluded above).

Ranges reflect a sweep across 4 topology styles (MobileNet, AlexNet, GoogLeNet, a 1-layer
sanity case) × 3 systolic array sizes (16×16/32×32/64×64) — the relative ranking of these
functions barely moves across that sweep, so this isn't a MobileNet-specific artifact.

## Why it's already hard to speed up further

`service_reads`/`manage_prefetches`/`check_hit` are **already O(1) per operation** — they used to
be linear scans and were rewritten to dict-based lookups in an earlier fixing round
(`PERFORMANCE_FIXES.md`, fixes #3/#4). What's left is pure CPython per-call overhead multiplied by
millions of calls (4.6M+ calls each, for just 3 layers), not a bad algorithm. There's no more
"replace a loop with a dict" move available here.

## Options for making it faster (real ones, no logging-related shortcuts)

1. **Batch multiple cycles per Python-level function call.** Right now `service_memory_requests`
   pays full Python call overhead once per simulated cycle. If the inner read/write servicing were
   restructured to process N cycles' worth of requests per call (vectorized over cycles, not just
   over addresses within one cycle), the call-count — and therefore the CPython overhead that
   dominates this profile — drops by a factor of N. Biggest potential win, also the most invasive
   change (touches the core cycle-by-cycle timing model).
2. **Cython / Numba-compile the innermost loop.** `service_reads`, `manage_prefetches`, and
   `check_hit` are tight, already-simple dict/set operations — good JIT/Cython candidates since
   there's no remaining algorithmic complexity to preserve, just interpreter overhead to remove.
   Lower risk than #1 (doesn't change the model's structure), but adds a build/compile dependency.
3. **Parallelize independent layers.** Each layer's simulation is currently run sequentially in
   `single_layer_sim`. Layers with no data dependency on each other (already true across e.g. a
   parameter sweep, or independent branches in a graph) could run as separate processes.
   Doesn't reduce total CPU time, but reduces wall-clock on multi-core machines. Not applicable
   within one strictly sequential layer-to-layer network unless the pipeline is restructured to
   overlap layers.
4. **Vectorize `service_writes`'s own logic** the same way `check_hit` was vectorized — smaller
   win (~5-6%) but same low-risk shape as the earlier fixes, and untouched by any of the logging
   work.

None of these touch trace/report generation — they're all about the cycle-by-cycle memory-request
servicing loop itself.

---

## How these numbers were produced (and how to reproduce them)

**Tool:** Python's built-in `cProfile`, driving `scalesim.scale_sim.scalesim` directly (no CLI
wrapper needed):

```python
import cProfile, pstats, io
from scalesim.scale_sim import scalesim

s = scalesim(save_disk_space=False, verbose=True,
             config='configs/scale.cfg',          # swap for a different array size
             topology='topologies/conv_nets/mobilenet.csv',  # or any topology CSV
             layout='layouts/conv_nets/test.csv',
             input_type_gemm=False)

pr = cProfile.Profile()
pr.enable()
s.run_scale(top_path='/tmp/profile_out/')
pr.disable()

ps = pstats.Stats(pr).sort_stats('tottime')   # 'tottime' = own time, excludes children
ps.print_stats(25)                             # top 25 functions by own time
```

`sort_stats('tottime')` is what makes this a "where does the time actually go" table rather than
a call tree — it ranks functions by time spent in their *own* code, not counting time spent inside
functions they call (that's `cumtime`, useful separately for seeing which top-level function owns
most of the run).

**To reproduce the specific percentages above:**
1. Use a small topology slice (first 2-3 layers of any `topologies/conv_nets/*.csv` — the header
   row plus N data rows) so a profiling pass finishes in seconds, not minutes. Full-network runs
   scale the same functions proportionally, just take longer to profile.
2. Clone `configs/scale.cfg` and edit `ArrayHeight`/`ArrayWidth` for different array sizes.
3. Run the script above once per (topology, config) combination, read off each function's `tottime`
   row, and divide by the wall-clock time printed by `pstats` at the top of the report
   (`N function calls ... in X.XXX seconds`) to get the percentage.
4. To separate "logging" from "core logic" functions by eye: anything that touches `savetxt`,
   writes to a `trace_mat`/`trace_matrix`, or is named `get_*_start_stop_cycles` is report/trace
   generation; everything else in the per-cycle path (`service_*`, `manage_prefetches`,
   `check_hit`, `calc_*_addr`, `create_*_mat`) is core simulation logic.

This is the same method `PERFORMANCE_FIXES.md` and `OPTIMIZATION_SURVEY.md` used — nothing new was
introduced here, this doc just re-runs it across more topology styles and array sizes and reports
only the non-logging half of the breakdown.
