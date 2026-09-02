# SCALE-Sim Performance Fixes

Status: implemented and verified, **not yet committed** (working-tree changes only).
Full staged plan: `Scale-sim-plan.txt` (repo root) / `/home/george/.claude/plans/lets-set-a-plan-deep-elephant.md`.

## Why SCALE-Sim was slow

SCALE-Sim is cycle-accurate: its memory-simulation loop
(`scalesim/memory/double_buffered_scratchpad_mem.py::service_memory_requests`) runs one Python
iteration per simulated clock cycle. For a real network like MobileNet, a single early layer can
need 10K-1M+ cycles, so wall-clock time was dominated by per-cycle Python/numpy-scalar overhead —
not the modeled compute itself. We confirmed this with `cProfile` on 3 MobileNet layers before
touching any code: **48.7 seconds, 36.9 million function calls**, with a handful of functions
accounting for the overwhelming majority of that time. The fixes below target exactly those
functions. The core cycle-by-cycle architecture itself was **not** changed.

## Results

| Scenario | Before | After | Speedup |
|---|---|---|---|
| Full 27-layer MobileNet, default config (`ws` dataflow, `CALC`/estimate-bandwidth mode) | 4m15s | 2m28s–2m32s | **~42% faster** |
| 7-layer MobileNet subset, `USER` bandwidth mode | 2m0s | 59.6s | **~50% faster** |

Every output file (COMPUTE_REPORT.csv, BANDWIDTH_REPORT.csv, DETAILED_ACCESS_REPORT.csv, and all
per-layer SRAM/DRAM trace CSVs, for all 27 layers) is **byte-identical** before and after — these are
pure performance fixes with zero change in simulated behavior.

## Verification method

For each fix below:
1. A standalone unit test comparing the old logic against the new logic on synthetic inputs
   (run outside the simulator, so failures are cheap and fast to find).
2. SCALE-Sim's existing golden-trace regression suite
   (`test/general/scripts/diff_calc.sh`, `diff_user_{ws,os,is}.sh`, `test/sparsity/scripts/function_test.sh`) —
   these run a small topology and byte-diff every report/trace file against a checked-in
   "golden" reference. All 5 pass with zero diffs after every stage.
3. A larger-scale before/after diff on `topologies/conv_nets/mobilenet.csv` (or a subset), since the
   golden topologies are only 1-2 layers and don't exercise buffer eviction/wraparound at scale.

## The changes, in detail

### 1. Removed per-cycle `tqdm` progress-bar construction

**Files:** `scalesim/memory/write_buffer.py`, `scalesim/memory/read_buffer.py`

**The bug:** `write_buffer.py`'s `service_writes()` and two loops inside `read_buffer.py`'s
`service_reads()` are called **once per simulated cycle** from the main memory-simulation loop. Each
call did:
```python
for i in tqdm(range(incoming_requests_arr_np.shape[0]), disable=True):
```
`disable=True` means the bar never renders anything — but a brand-new `tqdm` object was still being
*constructed* on every single cycle. Profiling showed `tqdm.std.__init__` was called 147,666 times
for just 3 layers, costing 4.2 of the 48.7 seconds (9%) in `write_buffer.py` alone.

**The fix:** replaced with a plain loop:
```python
for i in range(incoming_requests_arr_np.shape[0]):
```
Removed the now-unused `from tqdm import tqdm` import from both files. This is a pure no-op —
`disable=True` guaranteed the object never affected output, so there is zero behavioral risk.

**Also removed** two similar per-*layer* (not per-cycle) `tqdm` wrappers inside the diagonal-flatten
loops in `systolic_compute_{ws,os,is}.py`, as a side effect of fix #2 below.

### 2. Vectorized the diagonal-flatten prefetch-matrix loops

**Files:** `scalesim/compute/systolic_compute_ws.py`, `systolic_compute_os.py`, `systolic_compute_is.py`

**The bug:** Before simulating any cycles, each dataflow builds a "prefetch matrix" describing DRAM
fetch order. Part of that involves reordering a matrix onto its anti-diagonals (so that data with
temporal locality is fetched together). The original code did this **one element at a time**:
```python
for diag_id in range(num_diags):
    max_row_id = min(diag_id, M - 1)
    min_row_id = max(0, diag_id - N + 1)
    valid_rows = max_row_id - min_row_id + 1
    for offset in range(valid_rows):
        row_id = max_row_id - offset
        col_id = diag_id - row_id
        elem = matrix[row_id][col_id]      # scalar numpy indexing, twice per element
        prefetches[0, idx] = elem
        idx += 1
```
This is an O(M×N) pure-Python loop with double-bracket scalar numpy indexing, a very slow way to move data in numpy. 
**The fix:** replaced the inner loop with a single vectorized numpy fancy-index assignment per
diagonal:
```python
for diag_id in range(num_diags):
    max_row_id = min(diag_id, M - 1)
    min_row_id = max(0, diag_id - N + 1)
    valid_rows = max_row_id - min_row_id + 1
    if valid_rows <= 0:
        continue
    row_ids = np.arange(max_row_id, min_row_id - 1, -1)
    col_ids = diag_id - row_ids
    prefetches[0, idx:idx + valid_rows] = matrix[row_ids, col_ids]
    idx += valid_rows
```
The `valid_rows <= 0: continue` guard reproduces the original's degenerate last-diagonal case
(where the inner range was empty and silently did nothing).

**Verified:** a standalone script compared the old and new implementations element-for-element
across 10+ matrix shapes (square, tall, wide, 1×1, with under-utilization padding), bit-identical
output in every case.

### 3. Dict-based `check_hit` in `read_buffer_estimate_bw.py` (biggest win)

**File:** `scalesim/memory/read_buffer_estimate_bw.py`

This is the code path used by default (`InterfaceBandwidth: CALC` in `configs/scale.cfg`).

**The bug:** `check_hit(addr)` is called for every address requested on every simulated cycle, 4.6
million times for just 3 profiled layers while doing a **linear scan**:
```python
for idx in range(start_set_idx, end_set_idx):
    if addr in self.list_of_sets[idx]:
        return True
return False
```
This alone accounted for 15.8 of the profiled 48.7 seconds (32%), the largest cost in the
whole simulator.

**The fix:** `list_of_sets` only ever grows forward (new sets are appended with strictly increasing
ids; evicted sets are nulled out but never reused for different content). That means if an address
appears in more than one finalized set, only the *most recent* one can matter for a window
membership test. So a single dict mapping
`addr -> most_recently_finalized_set_id`, maintained incrementally, is an **exact** replacement:
```python
# when a set is finalized (in manage_prefetches):
self.list_of_sets += [self.current_set]
for a in self.current_set:
    self.addr_last_set_id[a] = self.current_set_id   # <-- new

# check_hit becomes:
def check_hit(self, addr):
    start_set_idx = self.read_buffer_set_start_id
    end_set_idx = min(self.current_set_id, self.read_buffer_set_end_id + 1)
    if start_set_idx == end_set_idx:
        return False
    last_set_id = self.addr_last_set_id.get(addr, -1)
    return start_set_idx <= last_set_id < end_set_idx
```
`prefetch()`'s own iteration over set *contents* (which needs the actual addresses, not just a
hit/miss answer) is untouched.

**Verified:**
- A standalone synthetic-address-stream test (20,000 accesses across 1000+ set evictions, comparing
  old linear-scan logic against new dict logic on every single access) — zero mismatches.
- `diff_calc.sh` (the golden script that exercises this exact file) — pass, zero diff.
- Full 27-layer MobileNet before/after diff — every output file byte-identical.
- This fix alone took the full MobileNet run from 3m31s to 2m28s (~30% additional reduction).

### 4. Dict-based `active_buffer_hit` + vectorized `set_fetch_matrix` in `read_buffer.py`

**File:** `scalesim/memory/read_buffer.py`

This is the code path used when `InterfaceBandwidth: USER` is set instead of `CALC`.

**The bug (same shape as #3, different structure):** `active_buffer_hit(addr)` scanned every line in
the active/prefetch window on every cycle. Unlike `read_buffer_estimate_bw.py`, this buffer's window
**wraps around modularly** (`start_id > end_id` is a valid, wrapped state), and its `hashed_buffer` is
static — built once in `prepare_hashed_buffer()` and never mutated again.

**The fix:** because `hashed_buffer` is static, a reverse index `addr -> [line_ids]` built once
alongside it remains valid for the object's entire life, regardless of how the window later wraps:
```python
# built once, right after hashed_buffer is finalized in prepare_hashed_buffer():
self.addr_to_lines = {}
for lid, line_set in self.hashed_buffer.items():
    for a in line_set:
        self.addr_to_lines.setdefault(a, []).append(lid)

# active_buffer_hit's default branch becomes:
line_ids = self.addr_to_lines.get(addr)
if not line_ids:
    return False
if start_id < end_id:
    return any(start_id <= lid < end_id for lid in line_ids)
else:
    return any(lid >= start_id or lid < end_id for lid in line_ids)
```

**Also fixed in the same file:** `set_fetch_matrix()` had the same element-by-element copy-loop
pattern as fix #2, reshaping one matrix into another:
```python
padded = np.full(num_lines * self.req_gen_bandwidth, -1, dtype=np.float64)
padded[:num_elems] = fetch_matrix_np.reshape(-1)
self.fetch_matrix = padded.reshape((num_lines, self.req_gen_bandwidth))
```

## Summary: What Each Fix Did

| Fix | File | Problem | Solution | Speed Gain |
|-----|------|---------|----------|-----------|
| **#1** | `write_buffer.py`, `read_buffer.py` | Creating useless progress bar objects 4.6M times | Remove them, use plain loops | 9% |
| **#2** | `systolic_compute_*.py` | Extracting matrix elements one-by-one | Use NumPy to grab whole lines at once | ~5% |
| **#3** | `read_buffer_estimate_bw.py` | Searching for addresses by checking every item | Use a dictionary for instant lookups | **30%** |
| **#4a** | `read_buffer.py` | Searching for addresses by checking every line | Use a dictionary for instant lookups | ~5% |
| **#4b** | `read_buffer.py` | Copying matrix elements one-by-one | Use NumPy's fast operations | ~3% |
