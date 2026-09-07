# COSMA on SCALE-Sim — Status

Quick-scan summary. For full reasoning/citations/numbers behind any line
here, see `ITERATION_HISTORY.md`.

## What we've built

**Pipeline** (`cosma/` layout: entry points at the top level, library modules in `helpers/`, docs in `docs/` — see `PIPELINE.md`)
- `run_cosma.py` — entry point: orchestrates graph → baseline sim → ILP solve → COSMA-aware sim → combined report (DRAM bytes, cycles, speedup), saves the occupancy plot by default
- `run_experiments.py` — entry point: batch sweep across models × budgets, auto-saved CSV, baseline cached once per model
- `visualize_spm.py` — entry point: fast, **SCALE-Sim-free** diagnostic: `--bounds-only` (instant M_R/MPMF), and a 2-panel PNG comparing baseline vs. COSMA SPM occupancy over time on the same byte-address scale
- `helpers/graph_builder.py` — parses `model.json` into `nodes`/`tensors` (COSMA-tracked activation tensors only; weights/bias/network-input excluded)
- `helpers/topology_builder.py` — `model.json` → SCALE-Sim topology CSV
- `helpers/baseline.py` — real SCALE-Sim driver: `run_baseline()` (plain, no COSMA) and `run_cosma_aware()` (COSMA-plan-driven, real engine numbers)
- `helpers/cosma_Ilp.py` — the ILP itself: Eq.1–12, fixed-schedule mode (paper's §III-E2); Eq.6/7 hold by construction since the schedule is fixed
- `helpers/model_resolver.py` — `.tflite` → `model.json` auto-export + cache, shared by `run_cosma.py` and `run_experiments.py`
- `toy_spill_model.json` / `toy_branching_model.json` — synthetic fixtures; the only graphs in the repo where a real spill/retrieve ever fires

**Engine modifications** (explicitly authorized: "mess with SCALE-Sim's codebase as long as it's still accurate simulation")
- `scalesim/memory/cosma_resident_buffers.py` — `CosmaResidentReadBuffer`/`CosmaResidentWriteBuffer`, narrow subclasses that only override the case COSMA already decided (asserted-resident tensor needs no fetch; a tensor's own creation needs no drain), delegating to unmodified SCALE-Sim logic otherwise. Regression-verified byte-identical to stock behavior when unused.
- `scalesim/memory/double_buffered_scratchpad_mem.py` — additive, backward-compatible param (`ifmap_buf_class`/`filter_buf_class`) to install the above

**ILP correctness**
- Implemented Eq.1–12 (memory allocation + tensor replacement) faithfully
- Found + fixed a real gap: `P`/`S` were completely unconstrained at the very first timestep (Eq.2/3's `if t > 0` guard had no base case), letting the solver plant a zero-cost phantom "preserved" tensor before it was even created. Caught via ResNet-50 (5/79 tensors affected), confirmed harmless to all previously-published numbers, fixed with an explicit base-case constraint
- Added `compute_structural_minimum_bytes()` (M_R) / `compute_mpmf_bytes()` (MPMF) — solve-free, instant feasibility bounds

**Validation methodology**
- Purpose-built toy graphs (incl. the persisted `toy_spill_model.json`) specifically to exercise spill/retrieve, since real models never do
- Found + fixed a real bug: a retrieve consumed by a non-conv layer (e.g. `ADD`) was silently charged as free
- Verified engine subclasses are byte-identical to originals when inactive
- Verified exact accounting identities against real SCALE-Sim runs (baseline − cosma = extra bytes, to the byte)
- Verified the "nothing rewards compact placement" claim against the actual paper PDF (Eq.12/17 — `L` never appears in the objective, only in constraints)

**Real-model results obtained** — MobileNetV2-CIFAR10, ResNet-20-CIFAR10, SqueezeNet-small-CIFAR100, Inception-V3, ResNet-50 (full ImageNet)
- `M_R == MPMF` exactly for all 5 — no budget exists for any of them where real spill/retrieve can ever fire
- Confirmed real, engine-simulated DRAM-traffic reductions (residency-driven, not replacement-driven) on all 5
- Diagnosed SCALE-Sim's slowness on ImageNet-scale models via profiling (single-core, pure-Python hot loop — not GPU/multi-core/RAM bound)
- Timed our PuLP/CBC solver precisely against the paper's Gurobi claim (sub-second only for the smallest models; 178s on Inception-V3)

## What's missing for a fuller match to the paper

- **Operator scheduling** — the other half of "Combined *Scheduling*..."; `C[a,t]` is fixed to `model.json`'s topological order, never a free variable. Single biggest scope gap vs. the paper's headline contribution.
- **Divide-and-conquer heuristic for NAS-scale graphs** (§IV) — not implemented, out of scope from the start of this work.
- **Gurobi** — using PuLP/CBC instead (same ILP semantics, meaningfully slower at scale: 178s vs. the paper's ~0.3s average on Inception-V3-sized problems).
- **The paper's own comparison baselines** — TensorFlow-Lite's linear allocator × {default, MPMF schedule} × {Belady, greedy replacement}. Not implemented. Our current baseline (SCALE-Sim's own default per-layer buffers) is a different, weaker comparison, so our % reduction numbers are not the same measurement as the paper's 84%/85%.
- **§III-E1 "Minimize Peak Memory Footprint" mode** (Eq.13–15) — not implemented. Needed to compute the paper's *true*, schedule-optimal MPMF; what we compute today is the peak under our one fixed schedule (upper bound on the real thing, though proven equal to `M_R` — the best any schedule could do — on all 5 tested models).
- **No NAS-style / wide-parallel-branch model tested** — all 5 real models are human-designed, mostly-linear-chain CNNs, exactly the class the paper itself says scheduling matters least for. We've never tested a graph shaped like the ones (DARTS, PNASNet, etc.) where the missing scheduling piece would actually be expected to bite.
- **Real-model spill/retrieve evidence** — mechanically implemented and verified correct, but only ever exercised on the synthetic toy fixture; zero evidence yet that it fires correctly at production scale (may simply reflect that streamlined human-designed CNNs rarely need it — an open question, not a known bug).

## Known caveats in how we're using/reporting results today (process, not ILP gaps)

- **Budget-matching**: baseline's real SRAM is fixed by `scale.cfg` (currently 64+64+64=192KB total) regardless of whatever `--budget-kb` is passed to COSMA — some of our comparisons handed COSMA more total memory than baseline ever had. Checked once (ResNet-20, 192KB vs. 512KB gave identical results) but not verified across the board.
- **Hardware-config governance**: `configs/scale.cfg`'s array size changed 16×16 → 64×64 mid-project (now committed) without the validated-results table being re-measured, silently invalidating direct comparison to `ITERATION_HISTORY.md`'s §2 table. No process yet to pin/version which config a given reported number used.
