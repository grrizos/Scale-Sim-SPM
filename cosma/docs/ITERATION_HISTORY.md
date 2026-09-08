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
`PYTHONPATH`, or from the repo root directly. (`cosma/` is organized as:
entry points — `run_cosma.py`, `run_experiments.py`, `visualize_spm.py` —
directly in `cosma/`; library modules only ever imported by those, in
`cosma/helpers/`; and docs, including this file, in `cosma/docs/`. See
`PIPELINE.md` for the full layout.)

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
| `--time-limit` | unbounded (`None`) | CBC solve time limit, seconds -- runs until CBC proves `Optimal`/`Infeasible`, however long that takes; pass a number to cap it and risk `Not Solved` instead |

### Running individual stages standalone

Useful when debugging one layer of the pipeline rather than the whole
thing:

```bash
# 1. Parse model.json, print a graph summary (node/tensor counts, skip-connections)
PYTHONPATH=..:. python3 helpers/graph_builder.py

# 2. Build the SCALE-Sim topology CSV, print the layer-id -> topology-row mapping
PYTHONPATH=..:. python3 helpers/topology_builder.py

# 3. Run SCALE-Sim per-layer in-process, print total compute cycles + DRAM bytes
#    (cross-check against SCALE-Sim's own COMPUTE_REPORT.csv / DETAILED_ACCESS_REPORT.csv if in doubt)
PYTHONPATH=..:. python3 helpers/baseline.py
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

### Seeing COSMA's algorithm actually work: `visualize_spm.py`

`run_cosma.py`'s summary only reports aggregate byte counts (residency
credit, spill/retrieve totals) -- it never shows *what COSMA actually
decided*: which tensor went where, when something was evicted, when
something stayed put. `visualize_spm.py` is the tool for that -- a direct
picture of the ILP's placement decisions (Eq.9-11's `L`) and its
tensor-replacement decisions (Eq.3/4/8's `S`/`R`), which are otherwise
invisible. Its `--bounds-only` mode (below) is a fast secondary use --
figuring out *whether a given model/budget has anything to show at all*
before rendering -- not its main purpose.

Like every command above except it never touches SCALE-Sim (no
`baseline.run_baseline()`, no `run_cosma_aware()`) -- it only loads the
graph and, if needed, solves the ILP, so it's fast even on ResNet-50/
Inception-V3-sized graphs. `--model-json` accepts a `.tflite` directly
too (auto-exported/cached via `model_resolver`, same as `run_cosma.py`/
`run_experiments.py` -- see item 27 below), with matching `--exporter`/
`--export-dir`/`--force-export` flags.

```bash
# Render the comparison diagram at one specific budget
PYTHONPATH=..:. python3 visualize_spm.py --model-json model.json --budget-kb 64
```
Saves a PNG to `cosma/spm_plots/` (or `--out <path>`) with two stacked
panels sharing a timestep x-axis *and* the same byte-address y-axis, so
bar heights are directly comparable between them. The bottom panel shows
COSMA's actual chosen SPM placement -- continuous blocks where a tensor
stays resident (this is where you *see* residency, mechanism #1/#2 in
the chat discussion this came from) at its real chosen address, explicit
spill (▽)/retrieve (△) markers where it doesn't (mechanism #4). The top
panel shows the baseline (no COSMA) regime using the *same* block style
sized by real byte count, but restacked from address 0 at every single
timestep and never carried to the next one -- since SCALE-Sim's real
engine never keeps anything resident across layers, there is no
placement to show, only "how many bytes were active right now" (every
timestep's stack is exactly `compute_structural_minimum_bytes()`'s own
per-node sum -- baseline's peak height across the whole chart always
equals the model's M_R). Every baseline block is also a DRAM fetch
(marked △), which is *why* it's redrawn from scratch every time, unlike
COSMA's blocks which persist. Also prints a plain-text per-tensor event
table (the diagram's data twin). Raises with the M_R/MPMF bounds baked
into the message if the budget is infeasible or the solve doesn't reach
`Optimal`, instead of a bare pulp status string.

**Important caveat, worth repeating every time this comes up**: for all 5
currently-exported real models, spill/retrieve never fires at any
feasible budget (M_R == MPMF for every one -- see Appendix item 19), so
the bottom panel on a real model will only ever show continuous blocks,
never a ▽/△ marker. `cosma/toy_spill_model.json` is a small synthetic
fixture built specifically because of this -- it's currently the *only*
graph in this repo where the diagram actually shows a spill/retrieve:
```bash
PYTHONPATH=..:. python3 visualize_spm.py --model-json toy_spill_model.json --budget-kb 0.1953125
```

```bash
# Fast secondary use: is there even a budget range where spill/retrieve
# *could* show up in the diagram for this model, before rendering anything?
PYTHONPATH=..:. python3 visualize_spm.py --model-json model.json --bounds-only
```
Prints the paper's `M_R` (structural minimum -- any budget below this is
provably `Infeasible`, no solve needed) and MPMF (the ceiling at/above
which spill/retrieve is always 0, so the diagram would be all continuous
blocks). If they're equal, don't bother rendering at different budgets
expecting to see something new -- there's nothing to show at any budget.

### Recipe: finding a real budget to sweep, instead of guessing

The two tools above chain directly into each other -- this is the
intended way to pick a `--budgets-kb` value for `run_experiments.py`
without manual bisection (a real problem hit early on: three guessed
budgets for ResNet-50 over SSH, none landing anywhere useful; see item 19
below for the full story that motivated building `--bounds-only` at all).

```bash
# 1. Get the floor (M_R) and ceiling (MPMF) -- instant, no SCALE-Sim,
#    works on a raw .tflite too (auto-exported/cached, same as the other
#    two entry points as of item 27 below).
PYTHONPATH=..:. python3 visualize_spm.py --model-json model.json --bounds-only
#   Structural minimum (M_R):   ...   bytes ( X.XX KB) at t=...
#   MPMF ceiling (0 spill at/above): ... bytes ( Y.YY KB) at t=...

# 2. Feed those numbers straight into run_experiments.py's --budgets-kb.
#    Anything below X.XX is guaranteed Infeasible -- don't waste a real
#    SCALE-Sim run finding that out again.
PYTHONPATH=..:. python3 run_experiments.py --models model.json \
    --config ../configs/scale.cfg --budgets-kb X.XX Y.YY
```

If `M_R == MPMF` (the case for all 5 real models tested so far -- see
item 19/§2), there is no budget at which spill/retrieve is ever nonzero;
one budget at/above `M_R` is enough to get a real, SCALE-Sim-verified
DRAM-reduction/speedup number, and any budget below it will fail fast
with `AssertionError: tensor N (X bytes) does not fit in the Y-byte SPM
budget` -- exactly the wall `--bounds-only` already predicted, just
confirmed against the real engine this time.

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
| `run_cosma.py` | Entry point: orchestrates the full pipeline, reports cycles/DRAM/speedup, saves the SPM occupancy plot by default |
| `run_experiments.py` | Entry point: batch runner, multiple models × budgets, table/CSV output |
| `visualize_spm.py` | Entry point: fast (no SCALE-Sim), ILP-only debug tool: `--bounds-only` prints M_R/MPMF instantly; otherwise renders a 2-panel PNG (baseline vs. COSMA SPM occupancy over time). Accepts `.tflite` directly (auto-exported via `model_resolver`, same as the other two entry points) |
| `helpers/graph_builder.py` | `model.json` → `nodes`/`tensors` dicts |
| `helpers/topology_builder.py` | `model.json` → SCALE-Sim topology CSV + layer-id↔row map |
| `helpers/baseline.py` | Per-layer SCALE-Sim simulation; `run_baseline()` (plain) and `run_cosma_aware()` (COSMA-plan-driven, real engine numbers) |
| `helpers/cosma_Ilp.py` | The ILP itself (Eq.1–12, fixed-schedule mode); also `compute_structural_minimum_bytes`/`compute_mpmf_bytes` (M_R/MPMF, solve-free) |
| `helpers/spm_allocator.py` | `SpmAllocator`/`SpmAllocationError` — live, byte-addressed replay of a solved plan during `run_cosma_aware()`, independently verifying it's physically realizable at the declared budget (no `scalesim` import; also usable standalone against the toy fixtures) |
| `helpers/model_resolver.py` | `resolve_model_json()` — `.tflite` → `model.json`, auto-exported + cached under `_exported/`; shared by all three entry points (`run_cosma.py`, `run_experiments.py`, `visualize_spm.py`) |
| `../scalesim/memory/cosma_resident_buffers.py` | `CosmaResidentReadBuffer`/`CosmaResidentWriteBuffer` — genuine zero-cost residency/creation in SCALE-Sim's own engine |
| `toy_spill_model.json` | Synthetic, verification-only fixture (4 tensors, 10-100 bytes) — a minimal genuine spill/retrieve demonstration |
| `toy_branching_model.json` | Synthetic, verification-only fixture — bigger/branchier (22 layers, two parallel inception-style blocks, 16-64KB tensors), also forces a genuine spill/retrieve |
| `spm_plots/` | PNGs saved by `run_cosma.py`/`visualize_spm.py` (gitignored, regenerable) |
| `_exported/` | Cache of `.tflite` → `model.json` exports made by `run_experiments.py` |
| `results/` | Timestamped CSV results from `run_experiments.py` (auto-saved by default) |
| `docs/cosma_integration_plan.md` | Original design doc (Phase-1 plan, predates this file) |
| `docs/STATUS.md` | Quick-scan bulleted summary: what's built vs. what's still missing to match the paper more fully |
| `docs/PIPELINE.md` | The mechanics: step-by-step flow of `run_cosma.py` (real, SCALE-Sim-verified) and `visualize_spm.py` (fast, ILP-only), with diagrams |

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
19. **Added a fast, SCALE-Sim-free budget-range finder and a baseline-vs-
    COSMA SPM occupancy diagram**, after manually bisecting budgets for
    ResNet-50/Inception-V3 over SSH proved "very insufficient" (three
    guesses — 9000KB=Infeasible, 10600/12000KB=Optimal/0-spill — never
    landed in the interesting range, and each guess cost a real SCALE-Sim
    baseline pass).
    - **`cosma_Ilp.compute_structural_minimum_bytes()`/`compute_mpmf_bytes()`**
      (new, purely additive) compute the paper's `M_R` and MPMF directly
      from `(nodes, tensors)` — no ILP build/solve, no SCALE-Sim. Below
      `M_R`, any budget is provably `Infeasible`; at/above MPMF, spill is
      always 0. Running both against every model in the repo confirmed,
      concretely, something only suspected before: **`M_R == MPMF`
      exactly for all five currently-exported real models**
      (MobileNetV2 60.00KB, ResNet-20 192.00KB, SqueezeNet 80.00KB,
      Inception-V3 8103.38KB, ResNet-50 9408.00KB) — there is no budget
      for any of them where nonzero spill/retrieve can ever happen. The
      ResNet-50 number lines up exactly with the empirically-observed
      9000KB=Infeasible / 10600KB=Optimal boundary above. This also
      confirms full ResNet-50 does complete under SCALE-Sim given enough
      wall-clock time (§2 above describes it as too slow to finish at
      all, based on an earlier, shorter attempt — it finished on a
      second machine once run to completion, just slowly; `Optimal`,
      76.21% reduction, 1.0355× speedup at 80MB).
    - **`cosma/toy_spill_model.json`** (new) — a small synthetic 4-tensor
      graph (`M_R`=200B, MPMF=210B) built specifically because no real
      model can demonstrate a genuine spill/retrieve. The first
      *persisted* toy fixture in this project (prior ones, e.g. item 5,
      were always hand-built in-memory objects in throwaway scripts).
      Documented in the file itself as synthetic/verification-only, not
      valid `baseline.py`/`topology_builder.py` input.
    - **`cosma/visualize_spm.py`** (new) — renders a 2-panel PNG per
      `(model, budget)`: top panel is the baseline ("no COSMA") regime,
      bottom is COSMA's plan. Both are derived structurally (no
      simulation): baseline's panel comes from a new
      `compute_baseline_resident_action()` that mirrors
      `cosma_Ilp.extract_results()`'s `resident_action` shape but purely
      from `producer_layer`/`consumer_layers` — every tensor shown as
      resident only at its own creation instant, refetched fresh at
      *every* consuming timestep, since SCALE-Sim's default engine has no
      cross-layer persistence at all (confirmed while building
      `cosma_resident_buffers.py` in item 15 — no hook exists anywhere
      for "already loaded by a previous layer"). This is also why the
      baseline panel's y-axis is a plain per-tensor row, not an SPM
      address — baseline never places anything into a shared address
      space to begin with (it uses three independent, fixed-size typed
      buffers), so drawing one would misrepresent it. The COSMA panel
      keeps the real byte-address y-axis the user explicitly asked for,
      built from `spm_plan` via a run-grouping algorithm (group a
      tensor's resident timesteps into contiguous same-address runs;
      proved, from Eq.1/3/4/11, that a spill always precedes any address
      change for the same tensor, so gaps and re-placement always
      coincide) with spill/retrieve marked as explicit status markers.
      Verified end-to-end on the toy fixture at both `M_R` (200B: real
      spill@t=1 + retrieve@t=3 to a new address, `extra_dram_bytes`
      totaling exactly 2×size as expected) and MPMF (210B: one
      continuous resident block, zero markers), and rendered cleanly
      against the real MobileNetV2 model at 64KB.
    - Also deleted `cosma/unified_spm.py` (a `UnifiedSPM` ledger stub —
      confirmed via repo-wide grep to be imported nowhere; superseded
      before it was ever wired in by `cosma_resident_buffers.py`'s real
      engine integration in item 15) and removed it from §6's table.

20. **Found and fixed a real ILP base-case gap (via ResNet-50), verified
    the "no placement reward" claim against the actual paper PDF, and
    redesigned the baseline panel to match COSMA's** — all from the user
    actually running `visualize_spm.py` on real models and asking "why is
    there so much empty space" and "are you sure nothing rewards compact
    placement, should I send you the paper."
    - **Verified against the primary source**: found the actual paper
      (`~/Downloads/Combined Scheduling, Memory Allocation (1).pdf`,
      arXiv:2311.18246) already on disk and read it in full rather than
      re-asserting from memory. §III-D Eq.12 (general) and §III-E2 Eq.17
      (the fixed-schedule mode we implement) both read exactly
      `argmin Σ_t Σ_a (S_{a,t}+R_{a,t})×Size(a)` — `L` (address) never
      appears in either objective, only in constraints Eq.9/10/11. Confirms
      the claim precisely: placement is unconstrained by cost once it's
      feasible, straight from the paper's own formulation.
    - **Found a real gap while explaining the ResNet-20@512KB "empty
      space" screenshot**: `cosma_Ilp.build_cosma_model()`'s Eq.2/3 loop
      only added the "must have been resident at t-1" constraint `if
      t > 0`, leaving `P[a, T[0]]`/`S[a, T[0]]` **completely
      unconstrained** for every tensor at the very first timestep — Eq.1
      alone doesn't forbid them. Confirmed concretely (not just in
      theory) on ResNet-50 @ 9408KB: 5 of 79 tensors (ids 35, 68, 92, 123,
      132) got a spurious `P` at t=0 in an `Optimal` solve, visible in the
      diagram as a nonsense cluster of blocks at the very left edge for
      tensors not actually produced until layers 14–53. Checked the
      S-side too (would have corrupted `extra_dram_bytes`, a real
      reported number, not just the diagram) — did not fire on ResNet-50
      or any of the 3 previously-validated models; those published
      numbers are confirmed unaffected. **Fixed** by adding explicit
      `P[a,T[0]]==0`/`S[a,T[0]]==0` base-case constraints (mirroring what
      Eq.2/3 already enforce for every other timestep, since there is no
      "t-1" before the first one for anything to have been resident at).
      Re-verified after the fix: ResNet-50 still `Optimal`, still 0
      spill, phantom entries gone; MobileNetV2/ResNet-20/SqueezeNet
      unchanged; `toy_spill_model.json` still spills/retrieves the same
      10 bytes each way (CBC landed on a *different* but equally-valid
      retrieval address in one case — itself a live confirmation of the
      point above: address really is a don't-care to the objective).
    - **Redesigned the baseline panel** at the user's request ("i want
      the baseline and cosma panel look similar to compare") — it used to
      draw isolated dots on an arbitrary per-tensor row index, which
      couldn't be visually compared to the COSMA panel's byte-address
      bars at all. New `visualize_spm.baseline_timestep_stacks()`: at
      each timestep, stack that timestep's active tensors (exactly
      `compute_structural_minimum_bytes()`'s own per-node set) from
      address 0 upward. The baseline panel now draws the *same* colored,
      byte-sized rectangles as the COSMA panel, sharing one y-axis scaled
      to `max(budget, baseline's own peak stack)` — the honest difference
      being a baseline block spans exactly one timestep and always
      restacks from 0 (nothing carries over — still no real placement,
      per the subtitle), while a COSMA block can span many timesteps at
      one fixed address because it's genuinely kept resident. This makes
      the comparison legible at a glance: e.g. on ResNet-20@512KB,
      baseline's tallest stack is exactly 196,608 bytes — precisely its
      M_R — while COSMA's blocks reach the full 512KB ceiling in its
      tightest region, directly showing COSMA trading budget headroom for
      duration of residency, something the old dot-based panel couldn't
      express at all.

21. **Explained the array-size/speedup mystery precisely (not just
    plausibly), added per-layer compute-vs-memory-bound logging to
    `run_cosma.py`, and built a bigger/branchier toy fixture.**
    - **The 64×64-vs-16×16 puzzle, resolved with real numbers, not a
      guess.** Item 20's finding that ResNet-20 got 0% speedup at 64×64
      despite a real 78.3% DRAM cut raised an obvious question: doesn't a
      *bigger* array mean *faster* compute, which should make memory
      matter *more*, not less? Checked directly (`baseline.run_baseline()`
      run twice, same model, 16×16 vs 64×64 configs): the 64×64 array
      really is 4.26× faster (208,501 → 48,983 total compute cycles) —
      bigger arrays are not slower. The actual mechanism is in
      `run_cosma.py`'s `_default_bandwidth_words_per_cycle()`: in `CALC`
      bandwidth mode (this project's mode throughout), SCALE-Sim has no
      single DRAM bandwidth number, so that function reuses the array's
      own width (`arr_col`) as the assumed DRAM bandwidth too — confirmed
      directly (16.0 words/cycle at 16×16, 64.0 at 64×64, exactly 4×).
      Since `dram_bytes` doesn't depend on array size but the assumed
      bandwidth does, `dram_bytes/bandwidth` shrinks by roughly the same
      factor compute does when the array grows, so whichever term
      (compute or memory) already dominated `max(compute_cycles,
      dram_bytes/bandwidth)` keeps dominating regardless of array size —
      it's an artifact of tying bandwidth to array width for estimation
      convenience, not a hardware fact (real DRAM bandwidth is a memory-
      interface property, independent of PE array size). At 16×16,
      compute (208,501) and the memory estimate (191,299) are close
      enough that individual layers actually flip to memory-bound,
      letting COSMA's DRAM cut shorten the real critical path (the
      measured 1.0299×); at 64×64 compute so thoroughly out-races even
      the *unoptimized* memory estimate (48,983 vs. 191,299) that no
      layer is ever memory-bound, so DRAM traffic literally cannot matter
      no matter how much of it COSMA removes. Checked whether `scale.cfg`
      supports an explicit, array-independent bandwidth (`InterfaceBandwidth:
      USER` + a fixed `Bandwidth` value, confirmed to exist in
      `scalesim/scale_config.py`) as a cleaner alternative to tuning array
      size — but `USER` mode routes to a different buffer class than
      `CALC` mode's `ReadBufferEstimateBw`, which `cosma_resident_buffers.py`
      is built specifically to subclass, so switching modes isn't a safe
      drop-in today; noted as a real, separate piece of future work rather
      than attempted here.
    - **Added per-layer compute-vs-memory-bound logging to `run_cosma.py`**
      (`layer_bound_breakdown`, `baseline_memory_bound_layers`,
      `cosma_memory_bound_layers` in the returned summary, plus a new
      verbose-mode print block) — exactly the visibility that was missing
      and had to be reconstructed by hand to explain the finding above.
      Verified against both configs: correctly reports 9/32 layers
      memory-bound at 16×16 (all 9 flip to compute-bound under COSMA) and
      0/32 at 64×64, matching the by-hand analysis exactly.
    - **`cosma/toy_branching_model.json`** (new) — a bigger, branchier
      synthetic fixture than `toy_spill_model.json`'s 4-tensor/10-byte
      example, at the user's request ("a lot of parallel paths... average
      size tensors, not 10 bytes"). 22 layers, two back-to-back
      inception-style blocks (3-way then 4-way parallel conv branches
      merging back together), each with its own long-lived skip tensor
      spanning the whole block; tensor sizes 16KB-64KB (int8, so
      `shape == bytes` directly), matching the real models' activation-size
      range rather than the original toy's 10-100 byte tensors. Verified:
      `M_R`=144.00KB, MPMF=208.00KB (a real 64KB gap, much wider than the
      original toy's 10-byte one); solved at 176KB and confirmed two
      genuine spill/retrieve events fire (tensors 112 and 115, each
      evicted and later retrieved at a *different* address) alongside two
      long-lived skip tensors (100, 109) that stay resident as one
      continuous block across their entire block's parallel phase.

22. **`run_cosma.py` now also saves the SPM occupancy diagram by
    default**, at the user's request after running it standalone and
    wanting the visual plan alongside the text report every time. New
    `save_plot`/`plot_out_path` params on `run_cosma()` (default
    `save_plot=False`, since `run_experiments.py` calls `run_cosma()` in a
    tight per-budget sweep loop and a plot per call there isn't wanted);
    the CLI turns it on by default, with `--no-plot` to opt out and
    `--plot-out` to override the path. Reuses the ILP solve already done
    inside `run_cosma()` (`cosma_Ilp.extract_results()`'s `result`) by
    calling `visualize_spm.compute_baseline_resident_action()` +
    `render_comparison()` directly — deliberately not a second CLI
    invocation of `visualize_spm.py`, which would re-solve the ILP from
    scratch and double the cost on anything Inception-V3-sized. Verified:
    reproduced the user's exact command (SqueezeNet @256KB), confirmed the
    PNG saves to the same default path `visualize_spm.py` itself would use
    (`squeezenet_small_cifar100_int8_tucker_svd_5_256kb.png`), and that
    `--no-plot` correctly skips it.

23. **Reorganized `cosma/` into `helpers/`/`docs/`/entry-points**, at the
    user's request to tidy the directory. Moved the four library-only
    modules (never run directly, only imported) into `cosma/helpers/`:
    `graph_builder.py`, `topology_builder.py`, `baseline.py`,
    `cosma_Ilp.py`, plus a new `helpers/__init__.py`. Moved all four `.md`
    docs into `cosma/docs/`: `ITERATION_HISTORY.md`, `STATUS.md`,
    `PIPELINE.md`, `cosma_integration_plan.md`. Left the three real entry
    points (`run_cosma.py`, `run_experiments.py`, `visualize_spm.py`) and
    all data files (`model.json`, both toy fixtures, `_exported/`,
    `results/`, `spm_plots/`) exactly where they were. Used `git mv` for
    every tracked file so history follows the move.
    - Fixed every import that broke: the three entry points now do
      `from helpers import graph_builder` (etc.) instead of bare
      `import graph_builder`; `helpers/baseline.py`'s own cross-imports of
      its new siblings became relative (`from .topology_builder import
      build_topology`, `from .graph_builder import compute_size_bytes`).
    - Fixed a real path-breakage risk, not just the imports: `baseline.py`
      and the `__main__` debug blocks in `graph_builder.py`/
      `topology_builder.py` all compute a `HERE`/`here` from their own
      `__file__` and use it to find `model.json` (and, for `baseline.py`,
      `../configs/scale.cfg`). Moving them one directory deeper would have
      silently pointed all of these at `cosma/helpers/` instead of
      `cosma/` — added one extra `os.path.dirname()` to each so they still
      resolve to `cosma/`, unchanged from before the move.
    - Verified end-to-end after the move: all `.py` files parse; both
      `visualize_spm.py` (no `PYTHONPATH` needed) and `run_cosma.py`/
      `run_experiments.py` (`PYTHONPATH=..:.` needed for `scalesim`) run
      correctly and reproduce previously-established numbers exactly
      (MobileNetV2 @64KB: 99,176/86,439 residency credit, 32.51%,
      1.0016×); both `helpers/graph_builder.py` and
      `helpers/topology_builder.py` still work as standalone debug
      scripts and still read/write `cosma/model.json`/`cosma/topology.csv`
      (not `cosma/helpers/`).
    - Updated path references in the docs that describe *current* usage
      (§3's standalone-debug commands, §6's file table) to the new
      `helpers/`/`docs/` locations, and added a directory-layout diagram
      to `PIPELINE.md`. Left the chronological Appendix narrative (this
      section) untouched, per its own "no content removed" convention —
      it describes what was true at each point in time.

24. **`run_cosma.py` now auto-exports `.tflite` inputs, and
    `run_experiments.py` now saves a full verbose log per (model, budget)
    combination.** Two related asks: `run_cosma.py` required an
    already-exported `model.json`, while `run_experiments.py` could take
    a raw `.tflite` directly; and `run_experiments.py`'s CSV only ever
    held the compact summary row, never the full per-run detail
    `run_cosma.py --verbose` shows.
    - **`helpers/model_resolver.py`** (new) — `resolve_model_json()`,
      moved out of `run_experiments.py` (which had it standalone) so both
      entry points share one implementation. Passthrough for an already-
      `.json` path; for a `.tflite`, exports via the trim project's
      exporter and caches the result under `--export-dir`, keyed by the
      input's last two path components. `run_experiments.py` now imports
      it instead of defining its own copy.
    - **`run_cosma()` gained `exporter`/`export_dir`/`force_export`
      params** (and the CLI, matching flags) — resolves `model_json_path`
      through `model_resolver.resolve_model_json()` right at the top of
      the function, before anything else touches it, so every downstream
      use (the ILP, both SCALE-Sim passes, the plot's default filename)
      already sees the real `model.json` path. Verified against a raw
      `.tflite` whose export was already cached from earlier sessions
      (`_exported/resnet50-tflite-float/resnet50.tflite` — hits the
      existing `_exported/_exported_resnet50-tflite-float/model.json`
      cache directly, confirming the naming derivation matches
      `run_experiments.py`'s exactly, byte for byte).
    - **`run_experiments.py` saves a full verbose report per combination**
      by default, under a timestamped `cosma/logs/run_<timestamp>/`
      directory (`--logs-dir` to override, `--no-logs` to skip) — the
      same timestamp the CSV uses, for easy correlation. New
      `_run_and_log()` wraps each `run_cosma.run_cosma(verbose=True)` call
      in `contextlib.redirect_stdout(io.StringIO())`, so the exact same
      print statements `run_cosma.py` alone would show get captured to a
      string instead of flooding the sweep's terminal output, then
      written to `<model>_<budget>kb.log` (reusing the same name-
      derivation logic as `visualize_spm.default_out_path()`). Also
      captures a failing combination's exception into its log rather than
      only the CSV's one-line error string.
    - **Caught and fixed a real bug during verification, not just a
      hypothetical**: the "Wrote per-combination logs to ..." message
      printed unconditionally whenever logging was enabled, even for a
      sweep where every budget failed the fast pre-check and `_run_and_log()`
      (the only place that actually creates the logs directory) was never
      called even once -- claiming logs were written when the directory
      didn't exist at all. Fixed by checking `os.path.isdir(logs_dir)`
      before printing the message. Verified both directions: an all-fail
      sweep (`--budgets-kb 1`) now prints nothing about logs, a normal
      sweep still does and the directory genuinely contains one `.log`
      file per combination with the exact same content `run_cosma.py`
      alone would print (verified byte-for-byte on MobileNetV2 @64KB).

25. **Log filenames now include the array size**, after the user asked
    how to be sure a run actually exercised SCALE-Sim and was pointed at
    "change `--config`'s array size and watch the cycle numbers move" as
    the strongest available proof (item 21's array-size/bandwidth finding
    already established this empirically) -- which immediately raised the
    obvious follow-up problem: re-running the same (model, budget) at a
    different array size would silently overwrite the previous log, since
    `_log_file_name()` only encoded model + budget. New
    `_array_dims_tag(config_path)` reads `ArrayHeight`/`ArrayWidth` via
    `scale_config.get_array_dims()` and returns e.g. `'64x64'`; folded into
    `_log_file_name()` as `<model>_<budget>kb_<array>.log`, computed once
    per `run_model_sweep()` call (not per budget) since it only depends on
    `config_path`. Verified directly: the same model+budget run through
    `configs/scale.cfg` (currently 16x16) and a 64x64 copy produced
    `model_64kb_16x16.log` and `model_64kb_64x64.log` -- two files, not
    one overwriting the other. (First verification attempt used a flawed
    test -- both configs accidentally ended up 16x16, since `configs/
    scale.cfg` had been reverted to 16x16 since item 21's 64x64 findings
    without that being noticed here -- caught by the log content itself
    showing identical numbers, redone properly with a genuinely different
    second config.)

26. **`run_experiments.py` now saves plots too (it never did before), and
    the array-size tag from item 25 was extended to plots and the results
    CSV as well**, so all three artifact kinds from one sweep -- CSV,
    logs, plots -- are tagged consistently and none of them silently
    overwrite a previous run at a different array size.
    - **`visualize_spm.default_out_path()` gained an optional `tag`
      param**, inserted before `.png` (`<model>_<budget>kb[_<tag>].png`).
      Deliberately *not* made to read the array size itself from a config
      path -- this module still never imports anything under `scalesim/`
      (see its own module docstring), so the tag stays a plain string the
      caller computes and passes in.
    - **`run_cosma.py` gained a local `_array_dims_tag(config_path)`**
      (mirrors `run_experiments.py`'s identical helper from item 25;
      small enough that a second copy was simpler than factoring out a
      shared module for two ~6-line functions) and now passes it to
      `default_out_path()` whenever `plot_out_path` isn't explicitly
      overridden -- so even a single standalone `run_cosma.py` run is
      protected from the same overwrite risk, not just sweeps.
    - **`run_experiments.py`'s `run_model_sweep()` gained `save_plots:
      bool = True`**, threaded into `_run_and_log()`'s `run_cosma.run_cosma()`
      call as `save_plot=save_plots` -- previously this was never passed at
      all, so `run_experiments.py` saved zero plots regardless of anything
      else. CLI: `--no-plots` to opt out, matching `--no-logs`'s pattern.
      Since `run_cosma()`'s own default naming is now array-tagged, this
      needed no naming logic of its own in `run_experiments.py` -- passing
      `save_plot=True` is enough; the "Saved SPM occupancy comparison to
      ..." line lands in that combination's log file (stdout is redirected
      there for the whole call), not the live sweep terminal.
    - **The results CSV filename also gained the array tag**:
      `results/run_<timestamp>.csv` -> `results/run_<timestamp>_<array>.csv`,
      computed once in `main()` (one `--config` covers the whole sweep, so
      this is well-defined at that level, unlike per-combination naming).
    - Verified end-to-end: a standalone `run_cosma.py` run saved
      `spm_plots/model_64kb_8x8.png` (config was 8x8 by the time this was
      tested -- confirms the tag reads the config live, not a cached
      value); a 2-budget `run_experiments.py` sweep on the same config
      produced both `spm_plots/model_{64,128}kb_8x8.png` (real files, ~200KB
      each, confirmed on disk, not just claimed in the log) and
      `results/run_<timestamp>_8x8.csv`.

27. **Found and fixed a real gap: `visualize_spm.py` couldn't accept a raw
    `.tflite` -- unlike `run_cosma.py`/`run_experiments.py`, it never
    called `model_resolver`, so passing a `.tflite` straight to
    `--model-json` hit `graph_builder.load_graph()`'s `json.load()` on a
    binary flatbuffer and crashed with `UnicodeDecodeError`.** Found while
    the user was working on a remote machine (`wil`) with only `.tflite`
    inputs on hand for Inception-V3 -- surfaced two separate problems in
    the same session, both worth recording:
    - **The actual gap**: `visualize_spm.py`'s `main()` called
      `graph_builder.load_graph(args.model_json)` directly. Fixed by
      importing `helpers.model_resolver` and calling
      `model_resolver.resolve_model_json(args.model_json, args.exporter,
      args.export_dir, args.force_export)` first, exactly like
      `run_cosma.py` does -- added matching `--exporter`/`--export-dir`/
      `--force-export` CLI flags, and switched the two other
      `args.model_json` call sites (`default_out_path()`, the plot title)
      to use the resolved path. `model_resolver.py` imports only `os`/
      `subprocess`/`sys` -- no `scalesim` -- so this doesn't break the
      module's documented "never imports anything under `scalesim/`, no
      `PYTHONPATH` needed" property; verified by re-running `python3
      visualize_spm.py --model-json <...>.tflite --bounds-only` with no
      `PYTHONPATH` set at all and confirming it still worked.
    - **A separate, non-bug gotcha along the way**: an earlier command on
      the same remote machine passed `--config` (and `--models`) as
      absolute paths copied from the local machine
      (`/home/george/Desktop/SCALE-Sim/configs/scale.cfg`), which don't
      exist on `wil` (`grizos@wil:/data/grizos/Scale-Sim-SPM`). Python's
      `configparser.read()` silently no-ops on a missing file instead of
      raising, so the failure surfaced many calls later as a confusing
      `configparser.NoSectionError: No section: 'general'` inside
      `scale_config.py`, not as a file-not-found at the actual mistake.
      Not a code bug -- documented as a caveat in `STATUS.md` instead,
      with the fix being to always pass paths relative to `cosma/` (e.g.
      `../configs/scale.cfg`) in commands meant to run on more than one
      machine.
    - Real numbers obtained this way for the full Inception-V3 model on
      `wil`: `M_R == MPMF == 8297856` bytes (8103.38 KB) at t=2
      (`CONV2D`) -- floor equals ceiling again, consistent with every
      other real model tested so far (item 19/§2), and explains the
      earlier `AssertionError: tensor 3 (2841728 bytes) does not fit in
      the 65536/131072/262144-byte SPM budget` errors at 64/128/256KB --
      those budgets were always going to fail, well below this model's
      true 8103.38KB floor.
    - Added the explicit "bounds-only -> feed into run_experiments.py"
      recipe to §3 above, since this was the first time the two tools
      were actually chained together end-to-end by a user rather than
      just documented as two separate, adjacent commands.

28. **Added a real, live, byte-addressed SPM allocator (`helpers/spm_allocator.py`)
    that replays every solved plan's Create/Preserve/Spill/Retrieve transitions
    during `run_cosma_aware()`, independently verifying physical realizability
    at the declared budget** -- not to get different numbers (a real, decisive
    finding along the way proved that's structurally impossible for a correctly-
    solved plan, see below), but because nothing in the pipeline previously
    verified the ILP's plan was actually physically realizable; it was simply
    trusted.
    - **How this started**: debugging a `wil` remote run (item 27) raised the
      question of whether `scale.cfg`'s SRAM sizing ever interacts with COSMA's
      `--budget-kb` at all. Investigation (two Explore agents, full mechanical
      trace) found it doesn't: `baseline.py`'s `_make_memory_system()` never
      reads `scale.cfg`'s SRAM fields -- every layer's buffer is sized to
      exactly that layer's own real tensor bytes, always, regardless of budget.
    - **A decisive finding that reframed the whole exercise**: under this
      project's actual `InterfaceBandwidth: CALC` config, the ifmap/filter read
      buffer class (`ReadBufferEstimateBw`) is *unconditionally stall-free by
      design* -- confirmed in SCALE-Sim's own source comment ("In estimate mode,
      operation is stall free"), and independently confirmed empirically: sweeping
      `IfmapSramSzkB`/`FilterSramSzkB`/`OfmapSramSzkB` from 1024KB down to 1KB on
      MobileNetV2 @64KB produced byte-for-byte identical cycle counts every time.
      Separately, COSMA's own Eq.9 already guarantees `sum(resident bytes at t)
      <= budget` for every t, and today's code already sizes every layer's own
      buffer to exactly its own real tensor bytes (the tightest sizing already
      possible) -- so a capacity-driven buffer-size clamp would, for any
      `Optimal` solve, provably never produce a smaller number than today's.
      Building that (originally planned, per the prior version of this session's
      plan file) would have been real effort for a proven no-op.
    - **The user's actual ask, once this was surfaced**: *"i need to have
      realistic behaviour of an spm data traffic with a fixed size, so cosma can
      calculate a plan for that size and a specific model, and scale-sim can
      execute that plan with a specific size of spm in order to map the traffic
      accurately."* Read correctly, this is about fidelity/independent
      verification, not different numbers -- `baseline.py`'s real SCALE-Sim loop
      blindly trusts `resident_action`'s per-tensor flags with zero live,
      cross-layer bookkeeping of its own. `SpmAllocator` is that missing check.
    - **Design**: `SpmAllocator.step(t)` replays one timestep's transitions
      against a live `{tensor_id: (address, size)}` map, called once per
      timestep from `_run_layers()`'s existing loop (including non-conv
      timesteps, which never reach `_simulate_layer()` but can still hold a
      resident tensor). Raises `SpmAllocationError` (carrying `.tensor_id`/
      `.timestep`/`.reason`) on any physical inconsistency: double-allocate,
      address collision, budget overflow, free/preserve without residency, or a
      preserved tensor's live address disagreeing with `spm_plan`. Also
      cross-checks the live occupant set against `spm_plan`'s own claimed
      resident set every timestep -- genuinely non-tautological, since
      `resident_action`/`spm_plan` are built by two separate loops over the same
      solved ILP variables in `cosma_Ilp.extract_results()`.
    - **A real bug found and fixed during verification, not just a hypothetical
      the design anticipated**: the first version only freed a tensor on an
      explicit `'S'` action. Running it against `toy_spill_model.json` (@200B)
      raised a false collision -- tensor 11 is `'P'` at t=2 and then has *no*
      entry at all at t=3 (its last consumer already ran; nothing ever retrieves
      it again), yet tensor 10's real Retrieve at t=3 correctly reuses tensor
      11's old address. This is optimal ILP behavior, not a bug in the ILP: Eq.12
      charges real bytes for an explicit Spill even if never retrieved, but
      charges nothing for just letting `P` lapse, so the solver has no reason to
      ever mark `'S'` for a tensor it will never retrieve again. **Fixed** by
      freeing anything no longer in `spm_plan`'s claimed resident set at each
      t -- explicit Spill or implicit lapse, uniformly -- before processing any
      Create/Retrieve at that timestep (ordering also matters for a second, real
      reason: the same fixture spills tensor 10 and creates tensor 11 into its
      freed address in the very same timestep, t=1).
    - **Verified end-to-end after the fix**: `toy_spill_model.json` (@200B) and
      `toy_branching_model.json` (@176KB) both replay cleanly via the standalone
      (SCALE-Sim-free) path with a genuine spill and retrieve each, zero
      `SpmAllocationError`s. Four deliberately-broken-plan cases (overlapping
      addresses, a `'P'` with no prior residency, an `'S'` for a never-resident
      tensor, and a real solved plan with one address mutated post-hoc) each
      correctly raised `SpmAllocationError` with the expected `.reason`. All
      three previously-validated real-model results were re-run through the
      full `run_cosma_aware()` path with a pinned config (matching the exact
      array size each was originally measured at, since the live `scale.cfg` was
      being actively edited by the user mid-session for unrelated experimentation)
      and reproduced byte-for-byte identical numbers with zero violations:
      MobileNetV2-CIFAR10 @64KB (99176/86439 credit, 32.5%, 1.0016x), ResNet-20-
      CIFAR10 @256KB (227632/2559421, 91.1%, 1.0299x), SqueezeNet-small-CIFAR100
      @96KB (499456/598888, 92.0%, 1.3599x) -- exactly confirming the Eq.9
      argument above: real verification now runs on every COSMA-aware
      simulation, and it changes nothing about the reported numbers.
    - **Known gap, not silently skipped**: no fixture exercises the full
      `run_cosma_aware()` -> allocator -> real SCALE-Sim path with an actual
      spill/retrieve happening together (real models never spill; the toy
      fixtures have no real conv params and can't run through `baseline.py`/
      `topology_builder.py` at all -- confirmed via their own `_comment` fields).
      Building a real-conv-param fixture that also forces a spill is a natural
      follow-up, not done here.
    - `run_cosma_aware()`'s signature gained three new *required* params
      (`spm_plan`, `tensors`, `memory_budget_bytes`) -- the one real call site
      (`run_cosma.py`) was updated; `run_experiments.py` needed no changes since
      it only ever calls `run_cosma.run_cosma()`, never `run_cosma_aware()`
      directly. `_make_memory_system()`/`_simulate_layer()` needed no changes at
      all -- the allocator gates what happens before they run, not what they
      compute.
    - **A second real gap found and fixed right after, while checking "does this
      have any output so I can tell it worked?"**: the `[COSMA SPM] verified N
      timesteps, peak occupancy X/Y bytes, 0 violations` summary line was
      originally gated behind `verbose`, and `run_cosma.py`'s call to
      `baseline.run_cosma_aware()` never actually passed `verbose` through at
      all (a pre-existing gap, not introduced by this change -- the same was
      already true for `_simulate_layer()`'s own SCALE-Sim-level verbosity), so
      the line silently never printed. First fix attempt (threading `verbose`
      through) surfaced a second problem: that same `verbose` flag also enables
      SCALE-Sim's own internal per-layer `tqdm` progress bars
      (`single_layer_sim.run()` -> `service_memory_requests()`) -- confirmed by
      running with it enabled: ~50 progress-bar lines for MobileNetV2's 64
      layers, which would bury the one line actually worth seeing. **Final
      fix**: made the summary print unconditional (not gated on `verbose` at
      all) instead of threading the flag through -- it's cheap (already-computed
      values) and is meant to be the default proof-of-verification signal,
      without opting into engine-level noise to get it. Verified: output is now
      23 lines (was 75 with the noisy version), zero `tqdm` bars, same numbers
      as before, `[COSMA SPM]` line present.

29. **Removed the default ILP solve time limits** after DenseNet-121 -- found
    this session as the first real (non-synthetic) model with `M_R != MPMF`
    (6328.25 vs. 8232.00 KB, per `visualize_spm.py --bounds-only`), unlike every
    other real model tested so far -- hit `RuntimeError: COSMA ILP did not solve
    to optimality: status=Not Solved` at 6900KB, a budget already independently
    confirmed feasible and inside that real spill/retrieve range. `Not Solved` is
    a distinct PuLP/CBC status from `Infeasible` -- it means CBC was cut off by
    its time limit before proving anything either way, not that no plan exists.
    distinct PuLP/CBC status from `Infeasible` -- it means CBC was cut off by its
    time limit before proving anything either way, not that no plan exists.
    `run_experiments.py`'s own `--time-limit` default was only 120s (lower than
    `run_cosma.py`'s own 360s CLI default -- an inconsistency that existed before
    this fix too), and a 121-layer, densely-connected model near its spill/
    retrieve boundary (the hardest region for a MIP solver to prove optimality
    on -- many close-to-tied candidate placements, unlike a budget with slack or
    one deep in `Infeasible` territory) can genuinely need much longer than any
    fixed default would guess. `cosma_Ilp.solve()` already defaulted
    `time_limit_sec=None` (unbounded) at the lowest level; both callers imposing
    a concrete override (`run_cosma.py`'s function default `120` and CLI default
    `360`; `run_experiments.py`'s CLI default `120`) were changed to `None`, so
    the unbounded behavior now actually reaches CBC by default -- run until
    `Optimal`/`Infeasible` is proven, however long that takes. The `--time-limit`
    flag still exists on both entry points for anyone who wants to cap it and
    accept `Not Solved` as a possible outcome instead. Verified MobileNetV2
    @64KB still solves instantly and reproduces the exact same validated numbers
    with no time limit passed at all.
