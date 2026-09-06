# COSMA + SCALE-Sim — Project Status & Iteration History

This is the living record of the COSMA↔SCALE-Sim integration: what's
built, how it works, what's validated, and what's still missing relative
to the COSMA paper (Li, Gupta, Malik — "Combined Scheduling, Memory
Allocation and Tensor Replacement for Minimizing Off-Chip Data Accesses
of DNN Accelerators", arXiv:2311.18246). `cosma_integration_plan.md` is
the original design doc; this file is what actually happened building
against it.

**Sections 1–6 are the current state** — read these to understand what
exists and how to use it. **The Appendix is the full chronological
iteration log** — the detailed reasoning, bugs found, and paper citations
behind every design decision, kept for when the "why" matters.

---

## 1. What COSMA does and how the pipeline actually works

A DNN's activation tensors need to live somewhere between layers. If the
on-chip scratchpad (SPM) is too small to hold everything at once, some
tensors must round-trip through DRAM — expensive, non-compulsory traffic.
COSMA is an ILP that decides, for a given SPM budget, exactly which
tensor sits where and when to spill/retrieve it, to minimize that
non-compulsory traffic. SCALE-Sim is the cycle-accurate simulator that
tells you what any of this actually costs in cycles.

**The pipeline is not "run SCALE-Sim once, then let COSMA improve it."**
COSMA's plan is computed independently of any cycle numbers, and
SCALE-Sim runs **twice**:

1. **`graph_builder.load_graph()`** — parse `model.json` into the tensor
   graph (sizes, producer/consumer relationships). No simulation.
2. **`baseline.run_baseline()`** — SCALE-Sim runs the **first** time,
   completely COSMA-unaware: every layer independently, no cross-layer
   memory sharing at all (SCALE-Sim's ordinary behavior). This is the
   "no COSMA" comparison point.
3. **`cosma_Ilp.build_cosma_model()` + `.solve()`** — COSMA's ILP runs.
   **This never sees SCALE-Sim's cycle numbers** — it only takes tensor
   sizes and the producer/consumer graph from step 1. Its objective is
   purely "minimize non-compulsory bytes moved given this SPM budget," a
   pure memory-placement decision made without any notion of compute
   cycles.
4. **`cosma_Ilp.extract_results()`** — pulls out `resident_action`: the
   concrete plan (which tensor is `C`/`P`/`R`/`S` at each timestep).
5. **`baseline.run_cosma_aware()`** — SCALE-Sim runs the **second** time,
   now fed that plan via `CosmaResidentReadBuffer`/`CosmaResidentWriteBuffer`
   (`scalesim/memory/cosma_resident_buffers.py`): wherever COSMA says a
   tensor is already resident (`P`), the fetch is genuinely simulated as
   free; every layer's own output write is genuinely simulated as staying
   on-chip. This gives the real "with COSMA" numbers.
6. **Combine** — `max(compute_cycles, dram_bytes / bandwidth)` per layer,
   comparing step 2's totals against step 5's.

**Real numbers vs. idealized numbers — the one thing to know before
reading any output.** Two genuinely different kinds of "COSMA benefit"
show up in the results, and they come from different places:

- **Residency credit** (`total_ifmap_residency_credit_bytes` +
  `total_ofmap_residency_credit_bytes`) — bytes SCALE-Sim's own engine
  (step 5) confirms are avoidable simply by keeping activations on-chip
  across layers, instead of SCALE-Sim's naive assumption that every layer
  independently refetches its input and drains its output to DRAM. This
  is **real, engine-simulated**, and it's usually the dominant effect —
  it's present even when nothing ever needs to spill.
- **Spill/retrieve overhead** (`total_idealized_spill_bytes`,
  `total_idealized_retrieve_bytes`, `total_real_retrieve_bytes`) — the
  cost COSMA pays to make things *fit* a tight budget. A retrieve
  consumed by a conv-like layer gets a real SCALE-Sim number; a spill, or
  a retrieve consumed by a non-simulated layer type (e.g. `ADD`), has
  **no SCALE-Sim analog at all** — there's no operator to attach a real
  demand matrix to — so it stays COSMA's own idealized tensor-size
  estimate. This is disclosed, not hidden, and it's usually zero (see §2).

The accounting always ties out exactly: `baseline_dram_bytes` −
(ifmap credit + ofmap credit) + (idealized spill + idealized retrieve) =
`cosma_dram_bytes`.

## 2. Current validated results

Every row below is `Optimal` at the smallest tested budget that clears
each model's structural minimum (the paper's own `M_R`: the maximum, over
all operators, of that operator's combined input+output size — see
§4/§5's note on tiling). All numbers are from real runs, not projections.

| Model | Budget | Status | Ifmap + ofmap residency credit | Spill / idealized-retrieve / real-retrieve | DRAM reduction | Speedup |
|---|---|---|---|---|---|---|
| MobileNetV2-CIFAR10 (float32) | 64KB | Optimal | 99,176 + 86,439 B | 0 / 0 / 0 | **32.5%** | 1.0016× |
| ResNet-20-CIFAR10 (float32) | 256KB | Optimal | 227,632 + 2,559,421 B | 0 / 0 / 0 | **91.1%** | 1.0299× |
| SqueezeNet-small-CIFAR100 (int8, Tucker-SVD) | 96KB | Optimal | 499,456 + 598,888 B | 0 / 0 / 0 | **92.0%** | 1.3599× |

None of these three models ever needs to spill/retrieve above its
structural minimum — every DRAM-reduction number above is 100% residency
credit. Two things worth understanding about this, not bugs:

- **These models are compute-bound** at the configured 16×16 systolic
  array. `max(compute_cycles, dram/BW)` is dominated by `compute_cycles`
  in almost every layer, so even a 90%+ DRAM reduction can translate to
  a small speedup (MobileNetV2: 1.0016×) or a larger one (SqueezeNet:
  1.36×) depending on how compute/DRAM-bound the specific model is.
- **Once a budget clears the model's MPMF** (minimum peak memory
  footprint — the paper's own §V-B2: *"given a memory budget of MPMF...
  COSMA can always eliminate the non-compulsory off-chip data
  accesses"*), spill/retrieve permanently stay at 0 and every larger
  budget gives numerically identical results — there's nothing further
  to optimize once every tensor can just stay resident the whole time
  it's needed. A genuine spill/retrieve tradeoff was only ever exercised
  on a small hand-built toy graph (Appendix, item 15) — none of the three
  real models tested are structured to need it above their minimum.

Two real models were found to be **out of the currently-tested range
entirely**, and correctly so, not as failures:
- **Full ImageNet ResNet-50** (102MB `.tflite`, 224×224 input): SCALE-Sim's
  own cycle-accurate simulation is too slow to finish in reasonable time
  at this resolution — a real SCALE-Sim characteristic, unrelated to
  COSMA's ILP, which was never even reached.
- **Full ImageNet Inception-V3**: has a 2.8MB single tensor, far above the
  128–256KB budgets tried. The paper itself never tests below `M_R` (the
  single-operator minimum) — see §5 below, this isn't a gap, it's outside
  COSMA's defined operating range by design.

## 3. How to run the simulation

All commands below assume the current directory is `SCALE-Sim/cosma/` and
`scalesim` is importable — either run from `cosma/` with the repo root on
`PYTHONPATH`, or from the repo root directly:

```bash
# from SCALE-Sim/cosma/
PYTHONPATH=..:. python3 <script>.py [args]

# equivalently, from SCALE-Sim/ itself
PYTHONPATH=. python3 cosma/<script>.py [args]
```

`pulp` must be installed once (`pip install pulp` — it bundles the CBC
solver, no license needed).

### Quick end-to-end run

```bash
cd SCALE-Sim/cosma
PYTHONPATH=..:. python3 run_cosma.py --budget-kb 64
```

Prints:

```
SPM budget: 65536 bytes
ILP status: Optimal
--- COSMA's contribution (real, SCALE-Sim-engine-simulated) ---
Ifmap bytes avoided by keeping activations resident ('P'): 99176
Ofmap bytes avoided (a layer's own output never leaves the chip at creation): 86439
--- COSMA's overhead to fit the budget (idealized -- no SCALE-Sim analog exists for these) ---
Idealized spill DRAM bytes (no SCALE-Sim analog): 0
Idealized retrieve DRAM bytes (consumed by a non-conv layer, no SCALE-Sim analog): 0
Real (SCALE-Sim-simulated) retrieve DRAM bytes: 0
Baseline DRAM bytes (no unified SPM): 570930
COSMA DRAM bytes: 385315
DRAM traffic reduction: 32.5%
Baseline total cycles: 152850.2
COSMA total cycles: 152604.0
Speedup: 1.0016x
```

`run_cosma.py` CLI flags:

| Flag | Default | Meaning |
|---|---|---|
| `--model-json` | `cosma/model.json` | path to the exported graph |
| `--config` | `configs/scale.cfg` | SCALE-Sim hardware config |
| `--budget-kb` | `128` | SPM capacity in KB |
| `--time-limit` | `120` | CBC solve time limit, seconds |

### Running individual stages standalone

Useful when debugging one layer of the pipeline rather than the whole
thing:

```bash
# 1. Parse model.json, print a graph summary (node/tensor counts, skip-connections)
PYTHONPATH=..:. python3 graph_builder.py

# 2. Build the SCALE-Sim topology CSV, print the layer-id -> topology-row mapping
PYTHONPATH=..:. python3 topology_builder.py

# 3. Run SCALE-Sim per-layer in-process, print total compute cycles + DRAM bytes
#    (cross-check against SCALE-Sim's own COMPUTE_REPORT.csv / DETAILED_ACCESS_REPORT.csv if in doubt)
PYTHONPATH=..:. python3 baseline.py
```

### Using `run_cosma.py` as a library (e.g. to sweep SPM budgets)

```python
import sys
sys.path.insert(0, 'cosma')   # or run with PYTHONPATH set instead
import run_cosma

for budget_kb in (64, 96, 128, 256):
    summary = run_cosma.run_cosma(
        memory_budget_bytes=budget_kb * 1024,
        ilp_time_limit_sec=90,
        verbose=False,
    )
    print(budget_kb, summary['status'],
          summary['total_ifmap_residency_credit_bytes'],
          summary['total_ofmap_residency_credit_bytes'],
          summary['total_idealized_spill_bytes'],
          summary['dram_traffic_reduction_pct'], summary['speedup'])
```

`run_cosma()` returns a dict with `status`, `baseline_total_cycles`,
`cosma_total_cycles`, `baseline_dram_bytes`, `cosma_dram_bytes`,
`total_ifmap_residency_credit_bytes`, `total_ofmap_residency_credit_bytes`
(the real contribution — see §1), `total_idealized_spill_bytes`,
`total_idealized_retrieve_bytes`, `total_real_retrieve_bytes` (the
overhead side), `dram_traffic_reduction_pct`, `speedup`, and the raw
`spm_plan`/`resident_action` from the ILP for per-timestep detail. Raises
`RuntimeError` if the ILP doesn't solve to `Optimal` (e.g. a budget below
the model's structural minimum — see §2's Inception-V3 note).

### Running multiple models/budgets: `run_experiments.py`

```bash
# one model, several budgets
PYTHONPATH=..:. python3 run_experiments.py \
    --models model.json --budgets-kb 64 96 128 256

# several models (raw .tflite gets auto-exported and cached under cosma/_exported/)
PYTHONPATH=..:. python3 run_experiments.py \
    --models model.json \
        /home/george/Desktop/trim/models/resnet20/cifar10/fp32.tflite \
        /home/george/Desktop/trim/models/squeezenet_small/cifar10/fp32.tflite \
    --budgets-kb 128 256
```

Prints a summary table and **always saves the full results as CSV** — by
default to a timestamped file under `cosma/results/` (`Wrote N rows to
cosma/results/run_<timestamp>.csv`). Pass `--out-csv <path>` to choose the
location, or `--no-save` to skip saving. A `.tflite` input needs
`--exporter` pointed at `trim/python_scripts/export_model.py` if it isn't
at the hardcoded default path. Every `(model, budget)` pair runs
independently — an infeasible budget or an unsupported op in one model
shows up as an `ERROR` row without aborting the rest of the batch, and
the SCALE-Sim baseline (budget-independent) is run at most once per
model, reused across every budget in its sweep.

### Testing `cosma_Ilp.py` in isolation on a toy graph

Useful when changing the ILP itself, to sanity-check a constraint change
in seconds against a known-correct small example rather than the full
64-layer graph — build a 3–4 tensor, 3–4 timestep `nodes`/`tensors` dict
by hand with `graph_builder.Node`/`Tensor` and pass it straight to
`cosma_Ilp.build_cosma_model()` (see Appendix item 5 for a worked
example).

## 4. What's implemented vs. the paper

| Paper section | Status |
|---|---|
| Eq.1–4 (create/preserve/spill/retrieve state machine) | ✅ implemented |
| Eq.5 (operator inputs must be resident) | ✅ implemented |
| Eq.6 (sibling tensors created together) | ✅ true by construction (fixed schedule) |
| Eq.7 (create exactly once) | ✅ true by construction (fixed schedule) |
| Eq.8 (spill at most once) | ✅ implemented |
| Eq.9 (fits within budget) | ✅ implemented |
| Eq.10 (non-overlapping placement via `u`/`d`) | ✅ implemented, incl. mutex |
| Eq.11 (address pinning via `V`) | ✅ implemented (`V = P`, proven equivalent) |
| Eq.12 (objective: minimize spill+retrieve bytes) | ✅ implemented |
| §III-F overlap-filtering to control ILP size | ✅ implemented |
| §III-E2 Fixed-Schedule mode (Eq.16–17) | ✅ this is what we built |
| §III-B/C free operator scheduling (the "Combined **Scheduling**..." part) | ❌ **not implemented** — see §5 |
| §III-E1 alternate objective (minimize peak memory footprint) | ❌ not implemented (different use case, not needed here) |
| §IV Divide-and-conquer heuristic for NAS-scale graphs | ❌ not implemented (out of scope per user) |
| Gurobi solver | ❌ using PuLP/CBC instead (same ILP semantics, slower solve) |

**Design constraint inherited from the paper, not a gap:** COSMA treats
each operator as an atomic unit and never tiles a tensor to make it fit
(§III-A: *"optimizing techniques used in mapping a single operator, such
as operator tiling, are not the focus of this paper"*). This means a
budget smaller than a model's `M_R` (the largest single operator's
combined input+output size) is always `Infeasible` by design — the paper
never tests below `M_R` either. See §2's Inception-V3 note for a concrete
case of this.

## 5. What's missing: operator rescheduling

Right now the layer execution order is fixed to `model.json`'s
topological order. COSMA only optimizes memory allocation and tensor
replacement on top of that fixed order — it never considers running the
layers in a different valid topological order, which is the actual
"Scheduling" half of "Combined Scheduling, Memory Allocation and Tensor
Replacement." To implement it, `cosma_Ilp.py` would need:

1. **`C[a,t]` becomes a real free binary variable** across all `t`, not a
   constant fixed to `producer_layer(a)`.
2. **Eq.6 restored as a real constraint** (siblings created together) —
   needed now because the ILP actually chooses when each node runs.
3. **Eq.7 restored as a real constraint** (create exactly once) — was
   trivially true before; must be explicit once `t` is free.
4. **Eq.5 evaluated across all `t`**, not just one fixed timestep.
5. **DAG-validity/precedence** — implied transitively through the `P`/`R`
   residency chain in the paper's Eq.1–12, but worth verifying empirically
   once `C` is free rather than assuming the encoding is airtight.
6. **`graph_builder.py`/`topology_builder.py` unaffected** — only which
   timestep each node is assigned to changes.
7. **`run_cosma.py` updated** to consume COSMA's chosen order for the
   SCALE-Sim re-run (currently reports against the fixed input order).
8. **Expect a real jump in solve difficulty** — `C` moving from a fixed
   diagonal (64 nonzero entries) to a full `64×64` binary block is exactly
   the paper's own `O(|T|×|A|²)` worst case (§III-F); their full
   formulation times out on complex NAS-style graphs without the
   divide-and-conquer heuristic.

**Expected payoff**: this model's structure (mostly a linear chain with
short residual skips, not DARTS/NASNet-style wide parallel branches) means
reordering likely won't change much here — worth confirming empirically
before investing further, rather than assuming either way.

## 6. File reference

| File | Role |
|---|---|
| `model.json` | Exported MobileNetV2-CIFAR10 graph (64 layers, 172 tensors) |
| `graph_builder.py` | `model.json` → `nodes`/`tensors` dicts |
| `topology_builder.py` | `model.json` → SCALE-Sim topology CSV + layer-id↔row map |
| `baseline.py` | Per-layer SCALE-Sim simulation; `run_baseline()` (plain) and `run_cosma_aware()` (COSMA-plan-driven, real engine numbers) |
| `../scalesim/memory/cosma_resident_buffers.py` | `CosmaResidentReadBuffer`/`CosmaResidentWriteBuffer` — genuine zero-cost residency/creation in SCALE-Sim's own engine |
| `cosma_Ilp.py` | The ILP itself (Eq.1–12, fixed-schedule mode) |
| `unified_spm.py` | Thin resident-tensor ledger (unchanged from original stub) |
| `run_cosma.py` | Orchestrates the full pipeline, reports cycles/DRAM/speedup |
| `run_experiments.py` | Batch runner: multiple models × budgets, table/CSV output |
| `_exported/` | Cache of `.tflite` → `model.json` exports made by `run_experiments.py` |
| `results/` | Timestamped CSV results from `run_experiments.py` (auto-saved by default) |
| `cosma_integration_plan.md` | Original design doc (Phase-1 plan, predates this file) |

---

## Appendix: Detailed iteration log

Full chronological history — the reasoning, paper citations, and specific
numbers behind every decision above. Read this when "why did we do it
this way" matters; §1–6 above are enough for "what exists and how do I
use it."

### Starting point

`cosma_integration_plan.md` described the target architecture but claimed
`graph_builder.py` and `model.json` were already done/provided. Neither
existed anywhere in the repo. `unified_spm.py`, `baseline.py`, and
`cosma_Ilp.py` existed as stubs — `cosma_Ilp.py` only had Eq.1–6 of the
paper's ILP, `baseline.py` shelled out to a `scale_sim.py` CLI and parsed
a `DRAM_access.csv` file that doesn't exist in this SCALE-Sim version.

1. **Sourced a real model instead of a synthetic one.** Found a real
   MobileNetV2 (width-multiplier 0.35) trained on CIFAR-10 in a sibling
   project (`/home/george/Desktop/trim/models/mobilenet_v2_a035/cifar10/fp32.tflite`),
   with a working exporter (`trim/python_scripts/export_model.py`) that
   produces exactly the schema COSMA needs. Exported it and copied the
   result into `cosma/model.json`: 64 layers, 172 total tensors, 64
   COSMA-tracked activation tensors, 10 skip-connection tensors — this
   matches `cosma_integration_plan.md`'s own numbers exactly, confirming
   the plan doc was originally written against this same export.

2. **`graph_builder.py` (new).** Parses `model.json` into `nodes`/`tensors`
   dicts. Each layer's `inputs` are split into `activation_inputs` vs.
   `weight_inputs` using `inputs_from` (`-1` = not produced by any layer:
   weights, bias, or the network's raw input — all correctly excluded from
   COSMA's tracked tensor set this way, no special-casing needed).
   Verified against the plan doc's own worked example: tensor 18, producer
   layer 5, consumers `[6, 9]` — reproduced exactly.

3. **`topology_builder.py` (new).** Emits a SCALE-Sim topology CSV from
   the 52 CONV2D/DEPTHWISE_CONV2D layers (SCALE-Sim doesn't simulate
   ADD/DENSE/REDUCE_MEAN — those get 0 cost). Two real gotchas fixed here:
   - SCALE-Sim's topology CSV has no padding field, so `SAME`-padded
     layers need their IFMAP H/W reconstructed as *already padded*
     (`_same_padded_dim`), or SCALE-Sim's own internal assertion
     (`Filter height cannot be larger than IFMAP height`) fails for the
     1×1 and 2×2 spatial layers deep in this CIFAR-10-sized network.
   - SCALE-Sim has no native depthwise-conv concept; followed the
     existing repo convention in `topologies/conv_nets/mobilenet.csv`
     (`Num Filter = 1` for depthwise rows).

4. **`baseline.py` (rewritten).** Drives `scalesim.single_layer_sim`
   in-process per layer (no subprocess, no CSV round-trip) — this is also
   the mechanism `run_cosma.py` would need later to inject a custom
   memory system. One non-obvious fix: SCALE-Sim's own `layouts.load_arrays()`
   crashes on an empty layout path even when custom-layout mode is off in
   the config — the working pattern (recovered from a deleted file,
   `git show a0d26e6:smm_scalesim_runner.py`) is to leave the `layouts`
   object unloaded entirely rather than pass it an empty/dummy file.
   **Validated**: total compute cycles (152,604) and total DRAM bytes
   (559,062) from `baseline.py` match SCALE-Sim's own official
   `COMPUTE_REPORT.csv`/`DETAILED_ACCESS_REPORT.csv` exactly, run
   independently through `scalesim.scale_sim.scalesim`.

5. **`cosma_Ilp.py` (completed).** Added Eq.7–12 (had only 1–6). Installed
   `pulp` (CBC solver, bundled, no license needed).
   - **Scope decision**: fixed the operator schedule to model.json's
     topological order (`t == layer id`, confirmed identical to
     `model.json`'s own `topo_sort.order`, which is the identity
     permutation for this graph) rather than treating it as free. This
     makes `C[a,t]` a constant instead of a decision variable, which
     satisfies Eq.6/Eq.7 by construction and removes `|A|×|T|` binary
     variables. **This is not an ad-hoc shortcut** — it's exactly the
     paper's own documented "Fixed Schedule" mode (§III-E2, Eq.16–17),
     used in their own evaluation as "COSMA FS."
   - Verified `V[a,t] = P[a,t]` (used as a plain alias in the code) is
     *exactly* equivalent to the paper's more elaborate AND-gate
     definition of `V`, not an approximation — Eq.2 already forces
     `P[a,t]=1 ⟹` the tensor was resident at `t-1`, which collapses the
     paper's `V >= res(a,t-1) + P(a,t) - 1` / `V <= res(a,t-1)` /
     `V <= P(a,t)` down to `V = P` in every case.
   - Overlap-filtered `u`/`d` variable pairs (only created for tensor
     pairs whose liveness windows actually intersect) — same mitigation
     strategy the paper describes in §III-F.
   - **Validated** on a hand-built toy graph: correctly forces a
     spill-then-retrieve when memory is tight (paid exactly 2× the
     tensor's size in non-compulsory traffic, as expected), correctly
     reports `Infeasible` for a genuinely too-small budget (single
     operator's live-tensor sum exceeds budget — matches the paper's own
     stated minimum-budget requirement in §III-D), and correctly reports
     `Optimal` with `extra_dram=0` once budget is generous. Also solved
     on the real 64-layer/64-tensor graph across several budgets in 1–4s.

6. **Found and fixed a real double-counting bug in the combination
   formula.** `baseline.py` runs every layer independently, so a layer
   consuming an activation tensor produced by an earlier layer *always*
   shows up as an "ifmap DRAM read" — SCALE-Sim has no notion of a shared
   SPM across layers. Since Eq.5 guarantees a COSMA-tracked activation
   input is resident at its consuming layer (via `P`, free, or `R`,
   already charged through `extra_dram_bytes`), charging SCALE-Sim's
   naive per-layer ifmap-DRAM-read on top of that would double-count the
   same transfer — and would make COSMA structurally incapable of ever
   showing a net DRAM reduction (it could only ever add overhead, never
   credit itself for keeping something resident). Fixed by splitting
   `baseline.py`'s DRAM output into `ifmap`/`filter`/`ofmap` components
   and waiving the ifmap component in `run_cosma.py` whenever a layer's
   activation input is COSMA-tracked.

7. **`run_cosma.py` (new).** Orchestrates graph → baseline → ILP → combine
   (per plan doc §3: `total_cycles = sum_t max(compute_cycles[t],
   (compulsory_dram[t] + extra_dram_bytes[t]) / BW)`), reporting cycles,
   DRAM bytes, and speedup vs. a plain SCALE-Sim baseline. Confirmed
   clean error paths for both infeasibility cases (single tensor too big
   for budget; budget big enough for any one tensor but not for a
   producer+consumer pair simultaneously resident).

8. **Checked the implementation against the actual paper text** (not just
   the plan doc's summary table) once the PDF was available. Eq.1–5,
   8–12 match exactly or via proven-equivalent simplifications (see
   above). Found one literal gap: the paper's Eq.10 includes
   `u[a,b,t] + d[a,b,t] <= 1` (mutual exclusion between "a above b" and
   "a below b"), which had been omitted. Added it — confirmed redundant
   in practice (both directions active simultaneously is already
   self-contradictory given nonzero tensor sizes; re-running after the
   fix produced identical results) but now matches the paper exactly
   rather than relying on that being true.

9. **`run_experiments.py` (new).** Batch runner on top of `run_cosma.py`
   for running multiple models and/or multiple SPM budgets in one
   invocation instead of hand-editing `run_cosma.py` calls. Takes either
   `model.json` paths or raw `.tflite` paths directly (auto-exporting the
   latter via trim's exporter, cached under `cosma/_exported/` so repeat
   runs skip re-exporting), prints a summary table, and can write CSV.
   Each `(model, budget)` combination is run in isolation — one
   infeasible/erroring combination is recorded as an error row rather
   than aborting the batch. **Validated** on two real models: the
   existing MobileNetV2-CIFAR10 graph, and ResNet-20-CIFAR10 exported
   fresh from `trim/models/resnet20/cifar10/fp32.tflite` — correctly
   reports `Infeasible` at 64/128KB (ResNet-20's structural minimum is
   higher, its largest tensor alone is already 64KB) and `Optimal` with
   7.44% DRAM reduction from 192KB up.

10. **Fixed: `run_experiments.py` results were silently lost if `--out-csv`
    wasn't passed.** The user ran a real batch (a compressed/quantized
    SqueezeNet-small-CIFAR100 variant, `int8`, Tucker-SVD rank 5, across
    64/96/128/256KB) without `--out-csv` and, once the terminal scrolled
    past it, had no way to recover the results — only the printed table
    had ever existed. Fixed by making `run_experiments.py` always save a
    CSV by default now, to a timestamped file under `cosma/results/`
    (`--out-csv <path>` still overrides the location; a new `--no-save`
    opts back out entirely). While fixing this, also caught and removed a
    leftover duplicate `print` statement from the original edit that
    referenced `args.out_csv` directly — harmless when `--out-csv` was
    passed, but printed a confusing `Wrote N rows to None` line whenever
    it wasn't (exactly the run that triggered this fix). Both the
    default-save path and `--no-save` were re-verified after the fix.

11. **Found real full-resolution ImageNet ResNet-50 tflites already in
    `_exported/`** and profiled a run against one (102MB `.tflite`, 79
    layers, 79 tensors, max single tensor 3.27MB). `graph_builder` stayed
    instant; `baseline.run_baseline()` (the real per-layer SCALE-Sim
    simulation) hadn't finished after 5+ minutes at a 16MB budget and was
    killed — SCALE-Sim's cycle-accurate simulation cost scales with each
    layer's actual operand-matrix size, and full 224×224 ImageNet layers
    on the small 16×16 array in `configs/scale.cfg` are genuinely slow to
    simulate cycle-by-cycle. This is a real characteristic of SCALE-Sim
    itself, unrelated to COSMA's ILP (which was never even reached).

12. **Fixed: a too-small budget wasted a full SCALE-Sim baseline run
    before failing.** The user tried a different ResNet-50 export at
    64/128/256KB — all three budgets are far below a single activation
    tensor's own size (602,112 bytes), so `cosma_Ilp.build_cosma_model()`
    was always going to hit its own `assert_tensors_fit_budget` check —
    but that check only ran *after* `run_cosma()` had already paid for
    the full (slow, per item 11) SCALE-Sim baseline pass, three times
    over, one per budget. Fixed two ways:
    - Extracted the check into a standalone
      `cosma_Ilp.assert_tensors_fit_budget(tensors, budget)` and moved
      the call in `run_cosma.py` to right after `graph_builder.load_graph()`
      — before the SCALE-Sim baseline call — so a hopeless budget fails
      in milliseconds instead of after minutes of wasted simulation.
    - Added an optional `layer_stats` parameter to `run_cosma.run_cosma()`
      to skip re-running SCALE-Sim when the caller already has it (SCALE-Sim's
      output doesn't depend on the SPM budget at all). `run_experiments.py`
      now resolves each model once, fast-checks every budget in the sweep
      *before* touching SCALE-Sim, runs the baseline at most once per
      model (only if at least one budget survives the fast check), and
      reuses it across every budget that does.
    - **Verified**: the same ResNet-50 case that previously hung for 5+
      minutes per budget now fails all three budgets in 0.195s total; a
      mixed sweep (one infeasible budget, three feasible ones, on the
      SqueezeNet model) now runs SCALE-Sim exactly once (confirmed via
      the "running SCALE-Sim baseline for ..." log line appearing only
      once) and produces byte-for-byte identical results to before the
      change.

13. **Connected SCALE-Sim's own memory model to COSMA's tensor sizes** (the
    user's core question: "how does COSMA know the size of the actual
    traffic and not the size of the original tensor, since the data of
    tensors will probably be tiled"). Investigated two candidate fixes:
    - *Make cross-layer residency a genuine SCALE-Sim "hit" (zero DRAM
      cost) instead of the analytic waiver in `run_cosma.py`.* Read through
      `double_buffered_scratchpad_mem.py` and `read_buffer.py` to check
      feasibility: SCALE-Sim gives every layer a **fresh, empty** buffer
      (a new `single_layer_sim`/`double_buffered_scratchpad` per layer,
      by design — this is precisely the "SCALE-Sim doesn't understand
      inter-layer memory" problem the whole COSMA project exists to work
      around). Making a tensor show up as already-resident across that
      boundary would mean patching SCALE-Sim's internal hit/miss and
      DRAM-access-counting logic directly. Ruled out at this point:
      `cosma_integration_plan.md` explicitly states *"We are NOT
      modifying SCALE-Sim's internal simulation engine"*, and
      hand-patching that machinery under time pressure risks silently
      corrupting cycle accuracy elsewhere. This piece stayed a documented
      analytic step — until item 15, where the user explicitly authorized
      modifying the engine and this got closed for real.
    - *Size SCALE-Sim's own per-layer buffers from real tensor data
      instead of `scale.cfg`'s flat 64KB/64KB/64KB defaults.* This one
      **is** safe and real: it uses only the officially-supported
      `single_layer_sim.set_memory_system()` hook (no internals touched),
      and it was a genuine, independent inconsistency — SCALE-Sim's own
      compute/tiling simulation for a layer never changed no matter what
      `--budget-kb` was passed to COSMA, because buffer sizing was
      completely disconnected from both the actual tensor sizes and
      COSMA's budget. **Implemented** in `baseline.py`: for every
      conv-like layer, `ifmap_buf_size_bytes`/`ofmap_buf_size_bytes` now
      come from the real activation tensor shapes in `model.json` (via
      `graph_builder.compute_size_bytes`), and `filter_buf_size_bytes`
      from the layer's real `weights.size + bias.size` — not a config
      guess. This is also model-consistent: COSMA's own ILP (Eq.9)
      already assumes a resident tensor is never partially/thrashed, so
      sizing SCALE-Sim's buffer to the *whole* tensor is exactly what
      COSMA's plan guarantees is true whenever it says a tensor is
      resident.
    - **Validated**: re-ran all three previously-tested models
      (MobileNetV2-CIFAR10, ResNet-20-CIFAR10, SqueezeNet-small-CIFAR100
      int8/Tucker-SVD). Compute cycles were unchanged for MobileNetV2
      (152,604 — expected, since its tensors were already smaller than
      the old flat 64KB buffers, so no stall-cycle difference from
      tighter, real sizing). DRAM byte totals shifted by a small, real
      amount (559,062 → 570,930 for MobileNetV2) now that buffer
      partitioning reflects actual data sizes. SqueezeNet (int8, so byte
      sizes scale very differently than float32) shifted more visibly
      (16.74% → 41.84% DRAM reduction at matching budgets) — checked this
      wasn't a bug (baseline `>` COSMA DRAM bytes throughout, no
      zero-size-buffer edge cases, numbers stayed internally consistent)
      before accepting it as a real, more physically-grounded result
      rather than noise.

14. **Closed the "extra_dram_bytes always uses idealized tensor size"
    gap from item 13**, after the user pushed back with three sharp
    questions: does COSMA tile tensors to make them fit (no — confirmed
    directly from Eq.9 and the paper's §III-A scope exclusion), how is
    changing SCALE-Sim's buffer size justified when a real SPM's total
    capacity is fixed (a fair critique of how item 13 was framed), and
    can COSMA's actual plan be handed to SCALE-Sim to emulate real
    traffic, compared against a true no-COSMA baseline in SCALE-Sim.
    Investigated `scalesim/memory/double_buffered_scratchpad_mem.py` and
    `read_buffer.py` in detail to find the honest boundary at the time:
    SCALE-Sim gives every layer **three independent, uncontended typed
    buffers** (ifmap/filter/ofmap SRAM) with no shared-capacity
    competition between them or across layers — it structurally cannot
    represent COSMA's *one* shared address space without modifying its
    internal engine (still off-limits at this point). What *was* correct
    and achievable: COSMA's ILP is already the right model of the shared,
    fixed-capacity SPM (Eq.9/Eq.10 guarantee everything simultaneously
    resident fits, by construction, whenever `solve()` returns
    `Optimal`) — SCALE-Sim doesn't need to re-derive that, it only needs
    to supply the real cost of a transfer *when a real transfer actually
    happens*.
    - **`cosma_Ilp.py`**: `extract_results()` now also returns
      `resident_action: {(tensor, timestep): 'C'|'P'|'R'|'S'}`, exposing
      which specific action was taken per tensor per timestep (previously
      only an aggregated `extra_dram_bytes: {timestep: bytes}` existed,
      with no way to tell which tensor or which action produced it).
    - **`run_cosma.py`**: rewrote the per-layer combination loop to pick,
      per component, between real and idealized numbers:
      - `'P'` (continuously preserved) → **0** DRAM cost, no SCALE-Sim
        event exists to consult, correctly free.
      - a layer's own output creation (`'C'`) → **always 0** in the
        COSMA scenario, unconditionally — proven by Eq.3
        (`S[a,t] <= C(a,t-1) + P[a,t-1]`): a tensor can only be spilled
        *after* being resident for a prior timestep, never at its own
        creation instant, so a freshly-created tensor always lands
        directly in SPM. (Previously the ofmap write was always charged
        in full in both scenarios — this was a second, undetected gap of
        the same shape as the ifmap one from item 6.)
      - `'R'` (just retrieved) → charge `baseline.py`'s **real**,
        already-tensor-size-driven `ifmap_dram_bytes[t]` for that layer,
        instead of COSMA's idealized `size(a)` — this is the direct fix
        for "how does COSMA know the size of actual traffic instead of
        the tensor size."
      - `'S'` (spilled) → **stays idealized** (`size(a)`), disclosed as
        a hard, irreducible limitation at this point: SCALE-Sim only
        ever simulates "this layer's own input fetch" / "this layer's
        own output write," never "evict some unrelated tensor sitting in
        memory right now" — there is no real event to substitute without
        modifying SCALE-Sim's engine (still off-limits here).
      - Filter (weight) DRAM: unchanged, always charged in full in both
        scenarios (weights are never COSMA-tracked, per the existing
        "activation tensors only" design decision).
      - The plain SCALE-Sim baseline ("the original boxes without
        COSMA," per the user's own phrasing) is unchanged: always the
        full, unwaived `ifmap_dram_bytes`/`ofmap_dram_bytes`, exactly as
        SCALE-Sim naively simulates each layer independently.
    - **Validated**: a hand-built toy graph (same one from item 5) with
      *mocked* `layer_stats` deliberately using different numbers than
      any tensor's own `size_bytes` (777 and 999 vs. the toy tensors'
      real size of 10) confirmed the substitution picks exactly the right
      source per action — `'P'` charged 0 despite the mock reporting 999
      for that layer, `'R'` charged the mock's 777 (not the idealized
      10). Re-ran all three real models: all still solve
      `Optimal`/`Infeasible` exactly as before at the same budgets, and
      DRAM traffic reduction increased substantially everywhere
      (MobileNetV2 17.4%→32.5%, ResNet-20 7.44%→91.1%, SqueezeNet
      41.8%→92.0%) now that the ofmap-write gap is also closed.
    - **`run_experiments.py`**: `RESULT_FIELDS`/`print_table` updated to
      report the new `total_idealized_spill_bytes` /
      `total_real_retrieve_bytes` split instead of the old single
      `total_extra_dram_bytes` column.

15. **User authorized modifying SCALE-Sim's own codebase** ("as long as
    it's still accurate simulation of what SCALE-Sim represents"),
    lifting the constraint items 13/14 had worked around analytically.
    Used this to close the two remaining gaps at the root rather than
    with more Python-side arithmetic.
    - **Investigated which memory classes are actually exercised.**
      `scale.cfg` has `InterfaceBandwidth: CALC`, so
      `double_buffered_scratchpad_mem.set_params()` instantiates
      `ReadBufferEstimateBw` (`scalesim/memory/read_buffer_estimate_bw.py`)
      for ifmap/filter, not the more complex hit/miss `read_buffer.py`
      investigated in item 13 — a different class than expected, found by
      re-reading the actual config path before designing anything.
      `manage_prefetches()`/`check_hit()` there group addresses into
      "sets" and only charge DRAM cost when a set fills and gets
      prefetched (`prefetch()`, `self.num_access += len(all_addresses)`)
      — confirming again there's no existing hook for "this data was
      already loaded by a separate `single_layer_sim` instance." The
      report getters `double_buffered_scratchpad.get_ifmap_dram_details()`
      (etc.) pull from `self.ifmap_buf.get_num_accesses()` /
      `.get_external_access_start_stop_cycles()`, both gated internally
      on a `trace_valid` flag — any override has to keep returning sane
      values there rather than letting them assert.
    - **`scalesim/memory/cosma_resident_buffers.py` (new).** Two
      subclasses, `CosmaResidentReadBuffer(ReadBufferEstimateBw)` and
      `CosmaResidentWriteBuffer(write_buffer)`, each overriding *only*
      the case COSMA's ILP has already resolved (a tensor asserted
      resident needs no fetch; a layer's own freshly-created output that
      stays resident needs no drain-to-DRAM, per Eq.3's proof that a
      tensor can only be spilled after being resident for a prior
      timestep, never at its own creation instant) — every other code
      path delegates unchanged to the real, unmodified parent class.
    - **`scalesim/memory/double_buffered_scratchpad_mem.py` (small,
      backward-compatible patch).** `set_params()` gained optional
      `ifmap_buf_class`/`filter_buf_class` parameters (default `None` =
      exact prior behavior for every existing caller) so a caller can
      install `CosmaResidentReadBuffer` in place of the default. The
      ofmap side needed no engine change at all — `set_params()` was
      already calling `self.ofmap_buf.set_params(...)` on whatever
      instance already lived there, so `CosmaResidentWriteBuffer` just
      gets pre-assigned before `set_params()` runs.
    - **Regression-verified both subclasses against the real originals**
      before using them anywhere: fed identical demand matrices through
      `ReadBufferEstimateBw`/`write_buffer` and through the new
      subclasses with the resident flag left `False` — byte-for-byte
      identical `get_num_accesses()` and `free_space` bookkeeping in both
      cases (9 vs 9 accesses; 60 vs 60; free_space 32 vs 32) — proving
      the override changes nothing about genuine fetch/write accuracy.
      With the resident flag `True`: 0 accesses, `free_space` fully
      reclaimed, cycles only advanced by `hit_latency` (i.e. exactly what
      "already on-chip" should look like).
    - **`baseline.py`**: refactored the per-layer simulation into a
      shared `_simulate_layer()`/`_run_layers()` pair, then added
      `run_cosma_aware(model_json_path, config_path, resident_action)` --
      identical to `run_baseline()` except it installs
      `CosmaResidentReadBuffer`/`CosmaResidentWriteBuffer` per layer,
      `ifmap_resident` set from `resident_action[(input_tensor, t)] == 'P'`
      and `ofmap_stays_on_chip` always `True`. Confirmed `run_baseline()`
      is unaffected by the refactor (152,604 cycles / 570,930 bytes,
      identical to item 13's numbers). On the real MobileNetV2 graph at a
      64KB budget, `run_cosma_aware()`'s ifmap total dropped to exactly
      3,267 bytes (only layer 0's raw network-input fetch — the one
      input that was never COSMA-tracked) vs. `run_baseline()`'s 102,443,
      and ofmap dropped to exactly 0 vs. 86,439 — both **real, simulated**
      numbers now, not asserted zeros.
    - **`run_cosma.py`**: reordered so the ILP solves before the SCALE-Sim
      pass (needed for `resident_action`), replaced the old
      action-based Python-side substitution logic with a direct call to
      `run_cosma_aware()`, and simplified the combination loop to sum the
      real per-layer numbers plus only the genuinely-idealized components.
    - **Found and fixed a real bug while validating on a purpose-built
      spill/retrieve test graph** (a small hand-authored `model.json`
      shaped like the item-6 toy example, run through the *actual*
      pipeline this time, not mocks): a retrieved tensor only has a real
      SCALE-Sim number to draw on when the layer consuming it is
      conv-like (the only kind `run_cosma_aware()` actually simulates).
      The test graph's retrieve happened to be consumed by an `ADD`
      layer — SCALE-Sim never simulates those at all, so
      `cosma_stats[t]['ifmap_dram_bytes']` was silently `0`, meaning a
      real retrieve was being charged nothing. Fixed by checking
      `nodes[t].op in ('CONV2D', 'DEPTHWISE_CONV2D')` before treating a
      retrieve as having a real number available; falls back to the same
      idealized `size(a)` treatment as spill otherwise (tracked
      separately as `total_idealized_retrieve_bytes` in the summary, so
      it's visible how much of the remaining traffic is real vs.
      idealized either way). Re-verified on the same test graph
      afterward: idealized spill 256B, idealized retrieve 256B (the
      `ADD`-consumed one), real retrieve 0B — correctly attributed.
    - **Final regression**: re-ran all three real models — results
      landed exactly on item 14's numbers (32.51%, 91.06%, 92.0% DRAM
      reduction respectively), which makes sense since none of them ever
      need to spill/retrieve above their structural minimum: the real
      engine-driven pass agreeing exactly with the earlier analytic
      approach, in the case where there's nothing for the two methods to
      disagree about, is itself a cross-validation that both are correct.

16. **User asked, looking at a CSV with `total_idealized_spill_bytes=0` and
    `total_real_retrieve_bytes=0` but a 91% DRAM reduction anyway: "what is
    COSMA's actual contribution and where do we see it in the logging?"**
    A fair question — once a budget clears the model's MPMF, spill/retrieve
    both stay permanently at 0 (see §2), but the dominant effect (avoiding
    SCALE-Sim's naive per-layer independent DRAM traffic entirely, via
    `CosmaResidentReadBuffer`/`CosmaResidentWriteBuffer` from item 15) was
    only ever visible by mentally subtracting `cosma_dram_bytes` from
    `baseline_dram_bytes` -- never printed or logged as its own number.
    Fixed: `run_cosma.py` now computes and reports
    `total_ifmap_residency_credit_bytes` (=
    `sum(base_s['ifmap_dram_bytes'] - cosma_s['ifmap_dram_bytes'])` across
    layers -- bytes avoided by keeping an activation resident via `'P'`)
    and `total_ofmap_residency_credit_bytes` (same, for the
    always-unconditional ofmap-stays-on-chip case) as first-class summary
    fields and verbose-output lines, labeled separately from the
    idealized spill/retrieve overhead lines (`--- COSMA's contribution
    ---` vs. `--- COSMA's overhead to fit the budget ---`).
    `run_experiments.py`'s `RESULT_FIELDS`/`print_table` updated to match.
    **Verified the accounting identity exactly**: re-ran ResNet-20 at
    256KB — `total_ifmap_residency_credit_bytes` (227,632) +
    `total_ofmap_residency_credit_bytes` (2,559,421) = 2,787,053, and
    `baseline_dram_bytes` (3,060,777) − 2,787,053 = 273,724 =
    `cosma_dram_bytes` exactly, with idealized spill/retrieve still 0 --
    confirming the new numbers are a correct decomposition of the
    already-validated total, not a new, separately-computed guess.

17. **User ran Inception-V3 (full ImageNet) at 128/256KB and got
    `Infeasible` on a 2.8MB tensor — asked whether COSMA should tile the
    tensor to make it fit, and what SPM sizes the paper itself uses.**
    Checked the paper directly rather than assuming: §III-D states
    outright that COSMA's minimum operable budget is *"the maximum over
    all operators of the sum of the sizes of the input and output tensors
    for an operator"* specifically *because* *"COSMA does not consider
    decomposition of a single operator (e.g., through operator
    tiling)"* — and §V-A confirms their own tightest tested budget, `M_R`,
    is defined as exactly that number; they never test below it. So the
    `Infeasible` result is correct, expected behavior matching the
    paper's own defined scope exactly, not a gap in this implementation
    — tiling is explicitly out of scope for COSMA as published. Folded
    this into §4 above as a documented design constraint (with the
    Inception-V3 case as the concrete example) rather than leaving it
    implicit across several conversation turns.

18. **Restructured this file for readability** at the user's request — the
    chronological iteration log (this Appendix) had grown to the point
    where the actual current state (what's built, how it works, current
    results) was buried inside dense blow-by-blow narrative. Added §1
    (conceptual "how the pipeline actually works" explanation, including
    the real-vs-idealized numbers distinction that came up repeatedly in
    conversation) and §2 (a single results table across all three
    validated models, plus the two out-of-range models and why) up front;
    moved the full chronological detail here. No content was removed —
    every fact, number, and citation from the original log is still here.
