# SMM Policy Integration — Performance Notes

Status: **fixed and verified**. See the "What was actually changed" section at the bottom for a
plain-language summary of the fix.
Companion doc for the general SCALE-Sim fixes: `PERFORMANCE_FIXES.md`.

## What the SMM integration does

`smm_scalesim_runner.py` + `smm_policy_selector.py` implement the SMM scratchpad-management
policies (Zouzoula et al., ICPP '24) on top of SCALE-Sim:

1. For every layer, `smm_policy_selector.best_for_layer()` evaluates 6 candidate policies
   (Intra, P1-P5 — ifmap-reuse, filter-reuse, per-channel, and their partial/blocked variants) and
   picks whichever minimizes off-chip accesses or latency (your `objective` choice), given a total
   on-chip GLB budget (`glb_size_kb`).
2. That policy determines the **exact minimum** number of bytes needed for the ifmap, filter, and
   ofmap scratchpad buffers — deliberately as small as the chosen access pattern allows, since
   minimizing on-chip memory footprint is the entire point of the algorithm.
3. `smm_scalesim_runner.py` injects those byte counts into a SCALE-Sim
   `double_buffered_scratchpad` instance (`_make_memory_system()`) before handing the layer to
   `single_layer_sim`, so the *real* SCALE-Sim memory simulation runs against SMM-managed buffer
   sizes instead of the config file's default (e.g. 64KB per operand in `configs/scale.cfg`).

The policy-selection math itself (`plan_network`, `best_for_layer`, `_evaluate`, `_choose_block`) is
pure Python arithmetic, runs once per layer before any cycle simulation starts, and is fast —
it is **not** the source of the slowdown described below.

## The problem

Running the same topology through `smm_scalesim_runner.py` is dramatically slower than running it
through plain `scalesim/scale.py` with the same array config — far more than the general per-cycle
overhead fixed in `PERFORMANCE_FIXES.md` can explain. Example: AlexNet's Conv1 layer alone, through
the SMM runner, took **22.5 seconds** in the memory simulation step (profiled with `cProfile`), where
the same layer through plain `scale.py` takes a small fraction of that.

### Root cause: small SMM buffers trigger a pre-existing O(n²) trace-accumulation pattern

Checked what buffer sizes SMM actually picked for AlexNet's Conv1 (`glb_size_kb=64`, policy P1):

| Operand | SMM-chosen | Default `scale.cfg` (64KB) | Ratio |
|---|---|---|---|
| ifmap | 7,392 B | 65,536 B | ~9x smaller |
| filter | 34,848 B | 65,536 B | ~1.9x smaller |
| ofmap | 5,184 B | 65,536 B | ~12.6x smaller |

Smaller buffers fill up and must drain to DRAM (or prefetch from DRAM) far more often. Every one of
those drain/prefetch events records itself into a running "trace matrix" using **repeated
`np.concatenate`**, which copies the *entire* trace accumulated so far on every single call:

```python
# scalesim/memory/write_buffer.py, append_to_trace_mat()  — runs on every buffer drain
self.trace_matrix = np.concatenate((self.trace_matrix, self.trace_matrix_cache), axis=0)
```
```python
# scalesim/memory/read_buffer_estimate_bw.py, prefetch()  — runs on every prefetch
self.trace_matrix = np.concatenate((self.trace_matrix, this_prefetch_traces), axis=0)
```
```python
# scalesim/memory/read_buffer.py, new_prefetch()  — runs on every prefetch (USER-bandwidth mode)
self.trace_matrix = np.concatenate((self.trace_matrix, this_prefetch_trace), axis=0)
```

If a trace is built from N drain/prefetch events, the total copying work across all of them is
`1 + 2 + 3 + ... + N ≈ O(N²)`, not `O(N)`. With SCALE-Sim's normal, generously-sized default buffers,
N is small (a handful of drains per layer) and this pattern is invisible. SMM's entire design goal is
the opposite — buffers as small as the access pattern allows — which makes N large and exposes the
quadratic cost.

**Profiled evidence** (AlexNet Conv1 only, through the SMM runner):

```
append_to_trace_mat        1,345 calls   5.65s own time    (write_buffer.py)
prefetch                     517 calls   1.52s own time    (read_buffer_estimate_bw.py)
manage_prefetches       3,296,484 calls  6.14s cumulative  (downstream of the small ifmap/filter buffers)
service_memory_requests        1 call   22.5s cumulative   (the whole per-cycle loop)
```
For comparison, the equivalent full 27-layer MobileNet run through plain `scale.py` (default 64KB
buffers, `Stages 1-4` fixes from `PERFORMANCE_FIXES.md` already applied) completes in ~2m28s total —
slower here on a *single* AlexNet layer through the SMM path.

### Why `PERFORMANCE_FIXES.md`'s fixes didn't already cover this

Stages 1-4 targeted the **per-cycle** hot path (`check_hit`, `active_buffer_hit`, the diagonal-flatten
prefetch loops, per-cycle `tqdm` construction) — all things that run once per simulated clock cycle
regardless of buffer size. This trace-accumulation cost instead scales with the **number of
drain/prefetch events**, which is normally tiny and only becomes large under SMM's small-buffer
policies. It's a different bottleneck, only reachable through this integration.

## Proposed fix (not yet applied)

Same principle in all three spots: stop concatenating onto a growing array on every event. Instead,
accumulate each chunk in a plain Python list and build the final array **once**, at the point where
the trace is actually read (`get_trace_matrix()` / `print_trace()`), e.g.:

```python
# instead of concatenating on every append_to_trace_mat() call:
self._trace_chunks.append(self.trace_matrix_cache)   # O(1) amortized

# and once, when the trace is actually needed:
self.trace_matrix = np.concatenate(self._trace_chunks, axis=0)   # O(N) total, done once
```

This preserves the exact same final trace content and ordering (same verification approach as
`PERFORMANCE_FIXES.md`: unit-test old vs. new accumulation on synthetic chunk sequences, then run the
golden-trace regression scripts) while turning the O(N²) accumulation into O(N).

## Open questions before implementing

- Whether to apply this to all three call sites (`write_buffer.py`, `read_buffer_estimate_bw.py`,
  `read_buffer.py`) or just the ones your SMM workflow actually exercises.
- Whether any code reads `self.trace_matrix` *mid-simulation* (before the layer finishes) rather than
  only at the end via `get_trace_matrix()`/`print_trace()` — if so, deferring the concatenation to
  "on first read" instead of "at layer end" would be needed to preserve exact behavior.

---

## What was actually changed

The problem in one sentence: three places in the code kept a running log of "what got sent to
memory, and when," and every time a tiny new entry needed to be added, the code copied the *entire
log so far* just to tack the new bit onto the end. Doing that a few times is nothing. Doing it
thousands of times — which is exactly what happens once your SMM policies shrink the on-chip buffers
— means you're re-copying an ever-growing log over and over, and that's where almost all the extra
time was going. It's like rewriting your whole diary from page one every time you want to add a
single new sentence, instead of just writing on the next blank page.

The fix is different depending on whether anything needs to *read* that log while the simulation is
still running, or only after it's completely done.

**`read_buffer.py` and `read_buffer_estimate_bw.py` (the two IFMAP/FILTER buffers):** nothing ever
looks at the log until the layer's simulation is fully finished — it's only read at the very end, to
write out a CSV report. So the fix here was easy: instead of copying the whole log every time,
just drop each new little piece into a simple pile, and only combine the whole pile into one final
log the *one time* it's actually needed. Same end result, none of the repeated copying.

The FILTER buffer had one extra wrinkle: sometimes a new piece is a slightly different width/shape
than the pieces before it, and the original code padded the shorter ones with filler so everything
lines up. We double-checked (with a written-out proof, then a test with hundreds of random-width
pieces) that padding everything to the same final width *once*, at the end, gives you the exact same
result as the original's "pad as you go" approach — so this piece got the same simple fix, just with
one extra padding step tacked on before the final combine.

**`write_buffer.py` (the OFMAP buffer): this one's trickier.** Here, the code *does* peek at the log
partway through the simulation — every time the buffer needs to empty itself out to make room, it
reads the log to figure out what to drain. So we can't just wait until the end to build it; the log
has to be correct and complete at every single point along the way. The fix here was to over-allocate
some room in advance and only make a fresh, bigger copy when we actually run out of space
(doubling the size each time), instead of resizing by exactly one small piece every single time.
That means the expensive "copy everything" step only happens a handful of times in total (each time
roughly twice as big as the last), instead of thousands of times — while the log itself is always
kept fully up to date and safe to read from at any moment, exactly like before.

None of these changes touch what ends up *in* the log or how the simulation itself behaves — only
how the log gets built along the way. Before trusting that, we checked it three ways: wrote small
standalone tests that compare the old and new logic side-by-side on made-up data (including, for the
OFMAP buffer, checking the log looks correct *after every single addition*, not just at the end, since
that's what the real code actually needs); reran SCALE-Sim's own built-in "does the output match
exactly" test suite; and reran the exact AlexNet scenario that was originally slow.

**Result:** that AlexNet layer through your SMM runner went from **26.2 seconds → 13.9 seconds**
(about **2x faster**), and every single output file it produces is byte-for-byte identical to before
the fix — same simulation, just without the wasted copying.

---

## Second issue found: crash on full-network runs (unrelated to the fix above)

Status: **fixed and verified**.

Running a full network (e.g. `topologies/conv_nets/mobilenet.csv`) through `smm_scalesim_runner.py`
with the default `glb_size_kb=64` crashes partway through, on layer 14 (`Conv15`):

```
ValueError: cannot reshape array of size 100352 into shape (0,16)
```

**This is a separate, pre-existing bug — not caused by the trace-accumulation fix above.** It sits in
different code (`set_params()` / `complete_all_prefetches()` in `read_buffer_estimate_bw.py`), which
none of the three fixes above touched. Confirmed by instrumenting the code and inspecting the crash
state directly.

### Root cause

`read_buffer_estimate_bw.py`'s `set_params()` computes:
```python
self.num_items_per_set = math.floor(self.total_size_elems / 100)
```
This assumes the buffer is always big enough that splitting it into "100 sets" gives at least one
element per set. For Conv15, SMM's `policy5 (partial-per-ch)` chose an ifmap buffer with
`total_size_elems < 100`, so this **floors to 0**.

With `num_items_per_set = 0`, the buffer's internal chunking (address requests are supposed to be
grouped into chunks of `num_items_per_set` and "finalized" once a chunk fills up) can never fire —
a chunk can never reach a target size of 0 once it has anything in it. So no chunk ever finalizes
during the whole layer, and every single distinct address requested (100,352 of them, in this case)
piles up into one never-closed chunk. That pile only gets handled by the end-of-layer cleanup method,
`complete_all_prefetches()` — which has its own bug: for this specific situation (nothing was ever
finalized mid-layer) it computes how many "cycles" of DRAM bandwidth are needed from leftover
placeholder cycle values that come out to exactly **0** here, then tries to fit all 100,352 real
addresses into a 0-sized array. That's the crash.

Confirmed by instrumenting `prefetch()` and printing state at the crash:
```
num_items_per_set: 0
current_set_id: 0
active_buffer_prefetch_done: False
cycles_needed: 0
len(list_of_sets): 1        <- everything ended up in one giant leftover chunk
```

### The fix

Gave `num_items_per_set` a floor of 1, so a buffer is never treated as having "0 items per chunk"
(`scalesim/memory/read_buffer_estimate_bw.py`, `set_params()`):
```python
self.num_items_per_set = max(1, math.floor(self.total_size_elems / 100))
```
With this, chunks finalize normally throughout the layer (as they do for every other, larger buffer),
and `complete_all_prefetches()` only ever needs to handle a small, normal-sized leftover chunk at the
end — the same code path that already worked correctly for every layer that didn't hit this edge case.

Worth noting: `read_buffer.py` (the `USER`-bandwidth counterpart to this class) computes the
equivalent value with `math.ceil` instead of `math.floor`:
```python
elems_per_set = math.ceil(self.total_size_elems / 100)
```
`ceil` of anything greater than 0 is always at least 1, so that class was never exposed to this bug
— only `read_buffer_estimate_bw.py`'s `floor` was the problem.

### Verification

- Golden-trace regression suite (`diff_calc.sh`, `test/sparsity/scripts/function_test.sh`) — zero
  diff. Default-sized buffers never hit the `< 100 elements` edge case, so this change is a no-op for
  every normal-sized buffer.
- Re-ran the exact crashing scenario — `smm_scalesim_runner.py` on
  `topologies/conv_nets/mobilenet.csv`, `glb_size_kb=64` — full 27-layer run now completes end-to-end
  with no error, including Conv15/Conv17/Conv19/Conv21/Conv23 (the `policy5 (partial-per-ch)` layers
  that were previously fatal).

---

## Third issue found: crash on GoogLeNet's FC6 layer (same root pattern, different file)

Status: **fixed and verified**.

Running GoogLeNet (`topologies/conv_nets/Googlenet.csv`, 60 layers) through `smm_scalesim_runner.py`
with `glb_size_kb=64` crashed on layer 57, `FC6` (`policy2 filter-reuse`):

```
File "scalesim/memory/write_buffer.py", line 291, in empty_drain_buf
    for elem in requests_arr_np[-1,:]:
IndexError: index -1 is out of bounds for axis 0 with size 0
```

**Same root pattern as the second issue above, but in `write_buffer.py` (the OFMAP buffer) instead of
`read_buffer_estimate_bw.py`, and unrelated to the trace-accumulation fix earlier in this doc** —
confirmed by instrumenting the code and by reverting the trace-accumulation fix, which didn't change
the crash (this arithmetic lives entirely in `set_params()`, untouched by that fix).

### Root cause

`write_buffer.py` splits every buffer into an "active" half and a "drain" half:
```python
self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
self.drain_buf_size = self.total_size_elems - self.active_buf_size
```
For `FC6`, SMM's `policy2` sized the OFMAP buffer down to **exactly 1 element total**
(`total_size_elems = 1`). With `active_buf_frac = 0.5`, `active_buf_size = ceil(1 × 0.5) = 1` — the
entire buffer — leaving `drain_buf_size = 0`. A buffer that can never drain anything can also never
free up space to accept new writes, so it deadlocks on the very first write, and the code crashes
trying to read a "next batch to drain" that can only ever be empty.

Confirmed by instrumenting `empty_drain_buf()` and printing state at the crash:
```
total_size_elems: 1
active_buf_size: 1
drain_buf_size: 0
trace_matrix.shape: (1, 32)
```

### The fix

Guaranteed the drain half is never zero when there's any capacity at all
(`scalesim/memory/write_buffer.py`, both `__init__` and `set_params()`):
```python
self.drain_buf_size = max(1, self.total_size_elems - self.active_buf_size)
```
A 1-element buffer genuinely can't be split 50/50 into two working halves — this trades exactness of
the requested `active_buf_frac` split for always leaving at least one element of drain room, which is
what actually lets the buffer make forward progress. It only changes anything for buffers this small;
every normal-sized buffer's split is unaffected.

### Verification

- Golden-trace regression suite (all 4 `diff_*.sh` + sparsity) — zero diff.
- Re-ran the exact crashing scenario — `smm_scalesim_runner.py` on
  `topologies/conv_nets/Googlenet.csv`, `glb_size_kb=64` — full 60-layer run now completes end-to-end,
  including `FC6` (previously fatal).
