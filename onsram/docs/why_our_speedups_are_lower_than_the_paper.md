# Why our OnSRAM speedups are much lower than the paper's

*Written 2026-09-24. Every number here was measured on this branch (`OnSram`)
with the paper config (`configs/scale_onsram.cfg`: 39×39 array, 32 B/cycle
DRAM, 2 MB SPM), unless it's marked as the paper's.*

> **Current state (2026-09-25):** all four causes below are fixed in
> OnSRAM's code. MobileNet now measures 1.36×, SqueezeNet1.1 1.24× and
> MobileNetV2 1.63×, each at 98–100% of our own infinite-SPM ceiling. What
> still separates us from the paper's headline numbers is explained in
> "Why compute takes so much of the execution time in our setup" near the
> end. In short: our graphs are fused (the paper's fused-layer result is
> 1.01–2.17×, avg 1.31×), and SCALE-Sim's array needs 1.6–2× the ideal
> compute time. The sections below keep the original analysis and
> numbers, with a "Status" note on each fix.

## The short version

On MobileNetV1 the paper reports that OnSRAM makes the network about **4.8×
faster**, and that an infinitely large SPM would make it **5.17× faster**
(Table 1). We measure **1.07×**.

The main reason is not the SPM-management logic. It's that **our simulator
thinks MobileNet is mostly limited by computation, while the paper's model
thinks it's mostly limited by memory traffic.** SPM management only reduces
memory traffic, so it can only help layers that are waiting on memory.

Measured directly: in our setup, even an *infinitely large* SPM, one that
removes all activation traffic, would make MobileNet only **1.28× faster**.
No pinning algorithm, however good, can beat that ceiling. The paper's
ceiling for the same network is 5.17×.

Four things contribute. In order of impact:

1. **Depthwise layers look extremely slow to compute in SCALE-Sim** (41–62×
   slower than the paper's kind of model). This is the dominant effect.
2. **Many tensors OnSRAM "pins" are not actually in the SPM when they're
   read.** A side effect of how this port implements the paper's Overwrite
   Optimization. 11 of MobileNet's 13 depthwise layers read their input from
   DRAM even though OnSRAM pinned it.
3. **Bytes are counted differently** (32-bit vs 16-bit numbers, re-reads).
4. **The free-output rule.** Earlier I listed this as a suspect for the gap,
   but that was wrong: it makes our OnSRAM numbers look *better*, not worse.
   It's a fairness problem, not the cause of the gap.

The rest of this doc explains each one in plain terms.

---

## Background: how both sides measure "how long does a layer take"

Think of a layer as a factory job. Two things can hold it up:

- **Compute time.** How long the machines (the systolic array) take to do the
  multiplications.
- **Memory time.** How long the delivery trucks (the DRAM link, 32 bytes per
  cycle) take to bring the inputs and weights in and take the outputs away.

These overlap (double-buffering), so the layer takes as long as **whichever
is slower**:

```
layer time = max(compute time, memory time)
```

This is the paper's own model (§6: "it estimates the execution time of a
node as max(compute_time, data_xfer_time)"), and **our port uses the same
formula** (`run_onsram.py`, `run_onsram_scale_sim()`: per layer,
`max(compute_cycles, dram_bytes / bandwidth)`). So the *formula* isn't the
difference. The difference is **what number goes into each side**:

| | Paper | Our port |
|---|---|---|
| Compute time | A model of a real accelerator, calibrated to within 1% of hardware measurements (§6, refs [93]/[10]) | SCALE-Sim's cycle-by-cycle simulation of a 39×39 systolic array |
| Memory time | Each input/weight/output element moved once ("ideal tiling… each data element is fetched once", §6) | The number of DRAM accesses SCALE-Sim actually generates (includes re-reads), ÷ 32 |

**What OnSRAM can change:** only memory time. Pinning a tensor in the SPM
means it doesn't cross the DRAM link. If a layer is already slow because of
compute, cutting its memory time changes nothing: `max(100, 50)` and
`max(100, 5)` are both 100.

So the whole question is: **for each layer, which side is bigger?**

---

## Cause 1: depthwise layers are very slow to compute in SCALE-Sim

### What a systolic array does well

The array is a 39×39 grid of multiply-add units. In this config
(weight-stationary dataflow), each **column** works on one output channel
(one filter), and the rows work through that filter's inputs. A normal
convolution with 64+ filters fills all 39 columns:

```
normal conv (64 filters)           depthwise conv (as SCALE-Sim sees it)
 col: 1 2 3 ... 39                  col: 1 2 3 ... 39
     [█ █ █ ... █]                       [█ · · ... ·]
     [█ █ █ ... █]                       [█ · · ... ·]
     [█ █ █ ... █]                       [█ · · ... ·]
   all 39 columns busy              1 column busy, 38 idle
```

### Why depthwise uses only one column

A depthwise convolution applies **one small filter per channel**, and
channels don't mix. SCALE-Sim's topology format has no way to say "one
filter per channel", so this port (following SCALE-Sim's own
`mobilenet.csv` convention) writes a depthwise layer as a normal conv with
**`Num Filter = 1`**. See `onsram/onsram_helpers/topology.py`. That gives
the right number of multiplications, but they all land in **one column**.
The other 38 sit idle.

### Measured

For each layer: `ideal` = FLOPs ÷ 3 TFLOP (every unit busy every cycle), and
`SCALE-Sim` = SCALE-Sim's own compute cycles (stalls removed):

| Layer | Ideal cycles | SCALE-Sim cycles | How much slower | Array utilization |
|---|---:|---:|---:|---:|
| CONV2D_2 (normal) | 16,890 | 25,317 | 1.5× | 66% |
| CONV2D_6 (normal) | 33,781 | 52,015 | 1.5× | 64% |
| **DEPTHWISE_1** | 2,375 | 101,271 | **42.6×** | **2.3%** |
| **DEPTHWISE_3** | 1,188 | 48,764 | **41.1×** | **2.4%** |
| **DEPTHWISE_13** | 594 | 37,008 | **62.3×** | **1.4%** |

Normal convolutions are close to ideal. Depthwise ones are 40–60× slower.
Across the whole network, depthwise layers are ~3% of MobileNet's
multiplications but **43% of our compute cycles**.

### Why that kills the speedup: a worked example (DEPTHWISE_1)

This layer reads a 112×112×32 input and writes a 112×112×32 output. That's
a lot of data for very little math, which is exactly the kind of layer SPM
management is built for.

**In the paper's kind of model** (compute near ideal, each element moved
once at 2 bytes):
- No SPM management: memory ≈ 50,000 cycles, compute ≈ 2,400 →
  **50,000 cycles** (memory-bound by ~20×).
- OnSRAM pins input and output: memory ≈ weights only (tiny), compute ≈
  2,400 → **~2,400 cycles**.
- **That one layer gets ~20× faster.** Layers like this are where
  MobileNet's 5× comes from.

**In our port:**
- No SPM management: memory ≈ 128,600 cycles, compute ≈ 101,000–118,500 →
  **~128,600 cycles**.
- Best case, input and output both in SPM: memory ≈ 20 cycles, but compute
  is still ≈ 101,000 → **~101,000 cycles**.
- **Best possible for that layer: ~1.3× faster**, because compute is now the
  bottleneck and SPM can't touch it.

### The ceiling, measured over the whole network

With every layer's activation traffic removed (infinite SPM, weights still
fetched), our MobileNet goes from 1,755,007 to 1,374,000 cycles: **1.28×**.
The paper's Table 1 says **5.17×** for the same experiment.

For the paper to get 5.17×, compute must be under ~1/5 of MobileNet's
baseline time in its model. For comparison, ideal FLOPs ÷ 3 TFLOP for all of
MobileNet is ~374,000 cycles, about 1/5 of our baseline time. So the paper's
result is consistent with an accelerator that runs depthwise layers
efficiently. (The paper's hardware, ref [10], also has a separate 375 GFLOP
SIMD array for non-matrix work. The paper doesn't say exactly how it runs
depthwise layers.)

**Bottom line:** our "real" systolic-array simulation is much more
pessimistic about depthwise layers than the paper's calibrated model. This
alone caps MobileNet at 1.28×, whatever OnSRAM does.

**Status: addressed (2026-09-24).** Both topology builders now write a
depthwise layer channels-across-columns (`Channels = 1, Num Filter = C`):
each column holds one channel's 3×3 filter. SCALE-Sim then times the
realistic mapping and gets filter and output traffic right. Because
SCALE-Sim feeds one input plane to every column, the runners replace the
depthwise input count with the real input tensor, each element read once.
Measured on MobileNet, paper config:

| | Before | After |
|---|---:|---:|
| DEPTHWISE_1 compute cycles | 101,271 | 12,658 (fold formula: 12,621) |
| DEPTHWISE_1 output traffic | 100,352 | 401,408 (the real tensor) |
| Baseline, whole network | 1,755,007 | 1,010,497 |
| Memory-bound conv layers | 12 of 28 | 18 of 28 (depthwise: 12 of 13) |
| Infinite-SPM ceiling | 1.28× | 1.18× |

The ceiling went *down*. Depthwise compute got faster in the baseline
too, and what's left is mostly pointwise-conv compute, which SPM can't
help. Depthwise layers are now memory-bound, but only by ~2× (the paper's
model: ~20×). The main remaining reason is Cause 3a: we charge 1 byte per
element where the paper's FP16 costs 2.

---

## Cause 2: "pinned" tensors that aren't in the SPM when they're read

### What the paper's Overwrite Optimization does

Say layer 1 reads tensor **A** (made by layer 0) and produces tensor **B**.
After layer 1, A is dead. The paper's Overwrite Optimization (§4.2) lets B
**reuse A's SPM space**: layer 1 reads A from the SPM and writes B into the
same space as A gets used up. Both are "on-chip" from layer 1's point of
view.

### What this port does instead

Our SPM checker (`SpmAllocator`) doesn't allow two tensors at the same
address at the same timestep, and our placement code can't express "B
overwrites A in place". So `pinning.py` makes **A leave the SPM one timestep
early**, right before layer 1 runs, to free the space for B. The
`pinning.py` docstring calls this "a conservative simplification".

The consequence: **layer 1 reads A from DRAM**, even though OnSRAM pinned A.

### Measured on MobileNet

The resident-action table (`'C'` = created in SPM, `'P'` = still in SPM):

```
t=0  CONV2D      output tensor 3:  {t0: 'C'}              <- leaves before t=1
t=1  DEPTHWISE   input  tensor 3:  (not in SPM at t=1)    -> read from DRAM
t=1  DEPTHWISE   output tensor 6:  {t1: 'C', t2: 'P'}
t=2  CONV2D      input  tensor 6:  'P' at t=2             -> read from SPM ✓
```

The same pattern repeats down the network. **11 of MobileNet's 13 depthwise
layers read their "pinned" input from DRAM.** (Of the other two, one input
was genuinely too big to pin and one is really in SPM.) 14 of the 27
"pinned" tensors are only in the SPM at the moment they're created.

This matters a lot: in our simulation the depthwise input read is the single
biggest DRAM cost (DEPTHWISE_1 reads 4.0M elements from DRAM for a 0.4M-
element tensor; see Cause 3).

It also creates a **physically impossible situation**. Because of the
free-output rule (Cause 4), tensor 3 is *never written to DRAM*, but layer 1
*reads it from DRAM*. It's read from a place it was never written to.

Fixing this would move us **toward** the paper and increase OnSRAM's
measured savings. But until Cause 1 is dealt with, the compute ceiling still
limits the speedup on depthwise layers.

**Status: fixed (2026-09-24).** `pinning.build_handoff_action()` now records
each hand-off read as a separate `'H'` ("read from SPM, then free") action.
It lives in its own map, so the shared `SpmAllocator` still sees A leave and
B take its space, but OnSRAM's `scale_sim_runner` treats the read as an SPM
hit. Measured on MobileNet @ 2 MB, paper config: 11 of 13 hand-off reads are
now served from SPM (the other 2 feed non-conv layers SCALE-Sim doesn't
run). The aware run's DRAM traffic dropped from 39.6M to 19.1M (baseline
51.1M). DEPTHWISE_1 now reads 0 input elements from DRAM (was 4.0M) and is
compute-bound at 101,271 cycles, as predicted above.

---

## Cause 3: bytes are counted differently

Three smaller differences in how data is counted, in the same spirit:

**a) Three different sizes per number.**
- The **SPM budget** (pinning decisions, `SpmAllocator`) uses model.json's
  **float32 = 4 bytes** per element.
- **DRAM traffic** comes from SCALE-Sim with `word_size=1`, so each element
  counts as **1 byte**.
- The paper's hardware is **FP16 = 2 bytes** (its SIMD unit is FP16; the
  text doesn't state tensor precision explicitly, so this is likely but not
  confirmed).

Consequences:
- Our tensors look **twice as big** to the 2 MB budget as they would at
  FP16. Fewer fit, so OnSRAM pins less than it would in the paper. Example:
  MobileNet's tensor 9 is 3.06 MB at float32 and couldn't be pinned; at FP16
  it's 1.53 MB and would fit.
- Our DRAM time charges **half** the paper's cost per element moved. That
  makes layers look less memory-bound than they would at 2 bytes, which
  again shrinks what SPM management can save.

**b) Re-reads.** The paper assumes every element crosses the DRAM link
exactly once. SCALE-Sim counts the accesses its dataflow actually makes:
- Depthwise layers read their input **~10× over**: DEPTHWISE_1 reads 4.0M
  elements for a 0.4M-element input, because each 3×3 window re-fetches
  overlapping pixels.
- Late pointwise layers write their output **~14× over**: CONV2D_14 writes
  1.4M elements for a 0.1M-element output, because partial sums are written
  back once per pass over 512 input channels ÷ 39 rows.

These inflate the *baseline's* memory time. On their own they'd make OnSRAM
look *better* (more traffic to save), but only for traffic OnSRAM actually
removes. With Cause 2, the biggest one (depthwise input) isn't removed.

**c) Stalls counted twice.** The "compute cycles" SCALE-Sim reports already
include its own memory-stall cycles (e.g. 17,234 of DEPTHWISE_1's 118,505).
The port then also takes `max(...)` against DRAM time, so some memory delay
is counted on both sides. It's a small effect.

**Status: fixed (2026-09-24/25), in OnSRAM's files only.** OnSRAM now
follows the paper's own accounting (§6):
- **a) One precision, FP16 = 2 bytes per element**
  (`scale_sim_runner.BYTES_PER_ELEMENT`), for the SPM budget (OnSRAM
  rescales its tensors in `run_onsram._at_paper_precision()`), buffer sizes
  and DRAM traffic. MobileNet's tensor 9 now fits and is pinned.
- **b) Fetch once.** Every layer's DRAM traffic is its real tensors, each
  element once: the input unless it's in the SPM, the output unless OnSRAM
  pinned it, the weights always. SCALE-Sim's own counts had been
  1.3–4.1× the fetch-once traffic, which pushed SqueezeNet to 3.48×,
  above the paper's own ∞ SPM bound.
- **c) No stalls in compute time.** Only SCALE-Sim's pure compute cycles
  are used (total minus its stall cycles); memory time comes from the
  traffic term of `max(compute, transfer)`.

SCALE-Sim now provides only compute time. COSMA still uses SCALE-Sim's
own counts, 1 "byte" per element, so COSMA and OnSRAM DRAM numbers are no
longer in the same units.

---

**How the free SPM room is modeled (2026-09-24).** Pinned tensors keep
their SPM space from layer to layer. The room they leave is each layer's
working area, split among ifmap, filter and ofmap (equal shares; an
operand that needs less keeps just what it needs). That split is **logged
only**. Every operand streams through with each element read once, which
is the paper's "ideal tiling" assumption (§6). Shrinking SCALE-Sim's read
buffers to the shares was tried and dropped: SCALE-Sim's buffer thrashes
when it's smaller than its operand. ResNet-50 layer 61 read its 2.36M
weights 118M times (50×), and a smaller buffer even gave less traffic than
a bigger one. That pushed ResNet-50 down to 0.55×.

## Cause 4 (corrected): the free-output rule. Unfair, but in OnSRAM's favour

### What it is

In the OnSRAM-aware run, the port marks **every** layer's output write as
free: no DRAM write, whether or not OnSRAM pinned that output
(`ofmap_stays_on_chip=resident_action is not None` in `scale_sim_runner.py`;
the log calls it "unconditional in this mode, not specific to OnSRAM's
pinning choices"). The baseline pays for every output write.

### What the paper does

Only **pinned** outputs stay on-chip. §3.2: an activation "can be pinned in
the on-chip SPM **as opposed to a write-back to external memory**". §3.1:
outputs "are in turn written back to the external memory". An unpinned
output costs a DRAM write.

### Why it doesn't explain the gap

It gives OnSRAM savings the paper wouldn't give, so it pushes our speedup
**up**. The gap goes the other way (ours is lower). Measured in the MobileNet
log: of the 21.2M bytes the aware run saves, **16.1M (76%) come from this
free-output rule** and only 5.1M from pinned inputs being read from SPM.
Charging unpinned outputs properly would make our number *lower*, further
from the paper, but more honest.

It's still worth fixing, because it's a fairness problem for the
three-paper comparison: it credits OnSRAM for something OnSRAM doesn't do.

**Status: fixed (2026-09-24), in OnSRAM's files only.** An output now stays
on-chip only if OnSRAM pinned it (`'C'` at its layer's timestep). Every
other output is written back to DRAM. COSMA keeps its unconditional rule,
because its own model (Eq. 7) creates every tensor in the SPM. The effect
is small because OnSRAM pins most outputs (MobileNet: 25 of 28), and the
remaining output-write saving is now legitimate.

## Results now (2026-09-25)

Paper config, 2 MB SPM, batch 1. Code state: hand-off reads from SPM,
channels-across-columns depthwise, ideal tiling, pinned-only free outputs,
FP16, fetch-once traffic, stall-free compute.

| Model | Ours | Our ceiling (infinite SPM) | Paper, fused layers (§7.1.6) | Paper, unfused (Table 1 / Fig. 7) |
|---|---:|---:|---|---|
| MobileNetV1 | **1.36×** | 1.38× | range 1.01–2.17×, avg 1.31× | 1-Step 2.84×, ∞ SPM 5.17× |
| SqueezeNet1.1 | **1.24×** | 1.24× | (same range, per-model not given) | 1-Step 1.57×, ∞ SPM 2.84× |
| MobileNetV2 | **1.63×** | | not in paper | not in paper |
| ResNet-50 | running | | | 1-Step 1.33×, ∞ SPM 1.75× |
| VGG16 | not run (out of memory on this machine) | | | 1-Step 1.03×, ∞ SPM 1.19× |
| Inception-v3 | fails in OnSRAM's placement step (pre-existing) | | | 1-Step 1.10×, ∞ SPM 1.64× |

Every run: 0 `SpmAllocator` violations. OnSRAM reaches 98–100% of our own
ceiling on every model, so the pinning itself is doing its job. What
limits the speedup is how much of each layer's time is compute, explained
next.

## Why compute takes so much of the execution time in our setup

### The one-line idea

SPM management only shortens **memory time**. A layer takes
`max(compute time, memory time)`, so if compute is already most of the
time, removing memory traffic can't help much. In our simulation, compute
is **78% of SqueezeNet's baseline time** and **69% of MobileNet's**. That
caps SqueezeNet at ~1.28× and MobileNet at ~1.44× (formula estimate; the
measured ceilings are 1.24× and 1.38×), however good the pinning is.

### What "compute time" is on each side

- **Our side:** SCALE-Sim simulates the 39×39 systolic array cycle by
  cycle, including how the layer is cut into tiles and how long each tile
  takes to fill and drain.
- **An ideal array:** `multiplications ÷ 1,521 units`, every unit busy
  every cycle. This is what "3 TFLOP" means on paper.
- **The paper:** compute time from a performance model calibrated to its
  real chip (§6, refs [10]/[93]). The paper doesn't print the formula, but
  its results only work if compute is a small part of the time (∞ SPM of
  5.17× on MobileNet means compute ≤ 1/5 of the baseline).

### Where SCALE-Sim's extra compute time comes from

A layer's weights form a table with **T rows** (kernel height × width ×
input channels) and **N columns** (output channels). The 39×39 array
handles a 39×39 piece of that table at a time (a "fold"), so a layer runs
as `ceil(T/39) × ceil(N/39)` folds, one after another. Each fold takes
`output pixels + 77` cycles. Two things make that slower than ideal:

1. **Half-empty folds.** The last fold in each direction is only partly
   filled, but still costs a full fold. Example: SqueezeNet's first conv
   has T = 3×3×3 = 27 rows, so 12 of the array's 39 rows sit idle for the
   whole layer; its 64 filters need 2 column folds, the second using only
   25 of 39 columns.
2. **Fill and drain.** Every fold spends ~77 extra cycles (39 + 39 − 1)
   pushing data into the array and results out, whatever its size. That's
   small for early layers with thousands of output pixels, but large for
   late layers with only 49 (7×7) or 169 (13×13): there the overhead is
   as big as the useful work, or bigger. MobileNet's classifier (1 output
   pixel, 27×26 = 702 folds) takes **81×** its ideal time. Batch size 1
   (the paper's setting) makes this worse: a larger batch would spread
   each fold's fill/drain over more pixels.

Measured totals (SCALE-Sim's fold formula, which matched measured cycles
to ~1% per layer earlier):

| | Ideal (every unit busy) | + half-empty folds | + fill/drain = SCALE-Sim | × ideal | Extra time from half-empty folds / fill-drain |
|---|---:|---:|---:|---:|---|
| SqueezeNet1.1 | 229,554 | 303,055 | 375,050 | 1.63× | 51% / 49% |
| MobileNet | 373,925 | 496,582 | 737,361 | 1.97× | 34% / 66% |

SCALE-Sim's actual pure-compute totals are another ~10–15% higher (411,000
and 856,159) from pipeline details the formula leaves out.

### How much this matters

With traffic counted exactly as now, the infinite-SPM ceiling becomes:

| | SCALE-Sim compute | Ideal compute | Paper ∞ SPM (unfused) |
|---|---:|---:|---:|
| SqueezeNet1.1 | 1.28× | 1.69× | 2.84× |
| MobileNet | 1.44× | 1.91× | 5.17× |

So compute explains part of the gap, but even an ideal array would leave
us well below the paper's headline numbers. The rest is the next point.

### The bigger reason: the paper's headline uses unfused graphs

The paper's main results run TensorFlow graphs where **BatchNorm, ReLU and
BiasAdd are separate nodes**. Each of those reads and writes a whole
activation tensor while doing almost no math, so they're extremely
memory-bound. That's exactly where OnSRAM wins (§7.1.3: "the activation-
bound layers such as ReLU, Pooling, and 'others' get pinned… the
performance benefits stem from the fact that the data transfer times for
these critical layers are drastically reduced").

Our models are TFLite exports where BatchNorm and ReLU are already
**fused** into the convolutions, and the remaining small ops (ADD, CONCAT,
pooling) aren't simulated at all (0 cost in both runs). So our graphs have
almost none of the nodes that give OnSRAM its big wins.

The paper measured this case too. §7.1.6: with layer fusion (Conv/MatMul
combined with BatchNorm-ReLU-BiasAdd), OnSRAM-Static achieves **1.01–2.17×,
average 1.31×**. That's the fair comparison for our fused models, and our
results (1.24–1.63×) sit inside that range.

### What we could still do

- **Compare against the paper's fused numbers** (1.01–2.17×, avg 1.31×)
  rather than Table 1. No code change; it's the like-for-like comparison.
- **Simulate the non-conv ops** (ADD, CONCAT, pooling) as memory-bound
  nodes (bytes in + bytes out, little compute). They currently cost 0,
  which hides some of OnSRAM's benefit.
- **Use ideal compute (FLOPs ÷ peak)** instead of SCALE-Sim's array, to
  mirror a roofline model. That would drop the cycle-level array
  simulation, and it only closes part of the gap (see the table above).
- Unfused graphs would need a different exporter (the reference
  implementation's `onsram_bw/` tree explored fused vs unfused).

---

## What this means, and what we could do

| # | Cause | Direction it pushes our OnSRAM speedup | Fixing it moves us… |
|---|---|---|---|
| 1 | Depthwise compute ~40–60× too slow | **down, a lot** (caps MobileNet at 1.28×) | closer to paper |
| 2 | Overwrite hand-off reads pinned inputs from DRAM | down | closer to paper |
| 3a | float32 SPM budget / 1-byte DRAM elements vs FP16 | down | closer to paper |
| 3b | Re-reads, partial-sum write-backs | mixed | closer to paper's "fetch once" |
| 4 | Every output write free | **up** | closer to paper (our number drops) |

Possible fixes, roughly in order of value:

1. **Cause 2 (hand-off).** Let the consuming layer read a reclaimed tensor
   from the SPM at its hand-off timestep. This is what the paper describes.
   It needs `SpmAllocator`/placement to accept "B overwrites A in place" at
   that one timestep. OnSRAM-only change, clearly paper-faithful.
2. **Cause 3a (one precision).** Use one bytes-per-element (FP16 = 2) for
   both the SPM budget and DRAM traffic. Worth confirming the paper's
   precision from ref [10]/[93] first.
3. **Cause 4 (free outputs).** Charge a DRAM write for outputs OnSRAM didn't
   pin. Once (1) is done, pinned-and-overwritten tensors stay consistent.
4. **Cause 1 (depthwise compute).** The hardest, because it's a question of
   *what we want to model*:
   - keep SCALE-Sim's systolic-array result (honest for this array, but not
     the paper's hardware);
   - model depthwise layers another way (e.g. running on a SIMD unit like
     the paper's, or using ideal FLOPs ÷ peak for them);
   - or report both.

   This affects all three papers the same way, so it should be one shared
   decision, not an OnSRAM-only patch.

Until (1) and (4) are addressed, **don't expect MobileNet-style networks to
get near the paper's numbers**. The best case in our current setup is 1.28×
for MobileNet, however good the pinning is.

---

## How these numbers were measured (to re-check them)

- Per-layer compute vs ideal: ran `single_layer_sim` on each MobileNet layer
  with `configs/scale_onsram.cfg` and the port's own `_make_memory_system()`,
  reading SCALE-Sim's compute report (total, stall) and detail report (DRAM
  counts).
- Infinite-SPM ceiling: same run over every layer. Baseline = Σ max(cycles,
  all DRAM ÷ 32); infinite SPM = Σ max(cycles − stalls, weight DRAM ÷ 32).
  The baseline total, 1,755,007 cycles, matches
  `onsram/logs/MobileNet_2MB_paper.log` exactly.
- Resident actions: `run_onsram.run_onsram()` on MobileNet at 2 MB, then
  looking up each conv layer's input tensor in `resident_action` at that
  layer's timestep.
- Paper quotes: from the PDF (`~/Downloads/OnSRAM_...rators-1.pdf`), §3.1,
  §3.2, §6, §7.1, Table 1.
