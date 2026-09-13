# OnSRAM + SCALE-Sim Integration — Project Plan

---

## 1. What We're Doing and Why

OnSRAM's own ideas are already implemented, independently of SCALE-Sim, at `/home/george/trim/spm_management/`. The job here is to port that logic into SCALE-Sim, the same way COSMA was ported, so OnSRAM gets real, cycle-accurate measurement instead of its current closed-form estimate. The two implementations stay **completely separate codebases** — different files, different algorithm, and this work never edits anything under `cosma/`.

**The paper**: "OnSRAM: Efficient Inter-Node On-Chip Scratchpad Management in Deep Learning Accelerators" — S. Pal, Swagath Venkataramani, Vijayalakshmi Srinivasan, K. Gopalakrishnan. ACM Transactions on Embedded Computing Systems, 2022 (DOI 10.1145/3530909). It extends an earlier 2021 IEEE ISPASS conference paper, "Efficient Management of Scratch-Pad Memories in Deep Learning Accelerators" — same authors, same idea, shorter version. (`cosma/docs/cosma_integration_plan.md`'s old line "NOT implementing OnSRAM or SPM-for-DL, those are separate papers" was referring to this conference/journal pair, not two unrelated papers.)

---

## 2. OnSRAM in Simple Terms

COSMA solves one big Integer Linear Program that jointly decides scheduling, placement, and eviction all at once — it's provably optimal but expensive to solve. OnSRAM takes the opposite approach: **fast, greedy heuristics, no solver**.

The paper has two variants. We're porting **OnSRAM-Static** (the one already built in `spm_management/`) — it works from a fixed, known computation graph, as opposed to OnSRAM-Eager, which handles models with no graph at all by guessing at runtime. OnSRAM-Static works like this:

1. **Score every tensor.** For each input/output tensor in the graph, look at when it's created, when it's last used, how many times it gets reused, how far apart those reuses are, and how big it is. Combine these into a single cost number — a Figure of Merit (FoM) — that estimates how expensive it would be to *not* keep this tensor on-chip (i.e. how much DRAM traffic would result from spilling it and reloading it later).

2. **Sort tensors by that score**, most expensive-to-spill first.

3. **Greedily pin them to the scratchpad, one at a time, highest score first — but only if the tensor can stay pinned for its *entire* lifetime** without blocking some other tensor that needs the space before this one is done being used. If it doesn't fit for its whole life, it's simply not pinned at all — there's no partial pinning, no evicting a tensor mid-life and bringing it back later. This is the single biggest conceptual difference from COSMA: COSMA can spill a tensor and retrieve it multiple times across a run; OnSRAM decides once per tensor, all-or-nothing.

Two more details worth knowing:
- Only activation tensors (inputs/outputs) are ever considered for pinning — weights are never pinned, since each layer's weights are only used once and have no reuse opportunity. Same convention COSMA already uses.
- The paper assumes **one single shared scratchpad** for the whole accelerator, not multiple separate memories — this matters for the port because it means COSMA's existing `SpmAllocator` (also a single unified budget) is a clean architectural fit, no adaptation needed there.

There's also a scheduling piece, confirmed directly against the real paper's §4.1 "BFS-DFS Schedule" (see §4 below): nodes are categorized as compute-bound, activation-bound, or weight-bound, and a node queue is walked BFS/DFS-hybrid style — when a committed node's children are enqueued, activation-bound children go to the *head* of the queue (scheduled as close as possible to their producer, minimizing liveness), while weight-/compute-bound children go to the *tail*. Realized in O(nodes+edges) time.

---

## 3. What Already Exists (`spm_management/`)

This is a **standalone, purely analytical implementation** — no simulator underneath it at all. It reads a JSON graph description (a subset of the same schema COSMA's `model.json` uses), runs the FoM/pinning logic above, and then estimates latency itself using closed-form formulas (FLOPs ÷ peak throughput, bytes ÷ bandwidth) rather than simulating anything cycle-by-cycle.

Two main script variants exist (`onsram_mine.py`, `onsram_weights.py`) plus a `onsram_bw/` variant tree exploring fused-vs-unfused graph handling and bandwidth sensitivity. **We're porting the core algorithm only** — the FoM scoring, greedy whole-interval pinning, and the scheduling step above — using `onsram_weights.py` as the reference (it's the more refined of the two: it adds an "Overwrite Optimization" that reclaims SPM space early when a tensor's producer overlaps with another tensor's death). The weight/activation SPM-congestion breakdown chart (`onsram_weights.py`'s other distinguishing feature) and the fused/unfused sensitivity study are explicitly **deferred to a later phase** — not part of this first port.

One loose end inside the reference file itself: its currently-active code path calls an older `calculate_fom()` heuristic, not the more paper-faithful `calculate_fom_corrected()` (which uses the paper's actual reuse-factor table based on kernel/channel dimensions) — that function exists in the file but isn't actually being called by anything today. **We'll port `calculate_fom_corrected()`**, since the goal is replicating the paper, not replicating the current code's dead-code gap. Worth double-checking this choice if the numbers don't line up the way expected.

---

## 4. Checked Against the Real Paper
Paper location: home/Downloads/OnSRAM_ E_icient Inter-Node On-Chip ScratchpadManagement in Deep Learning*.pdf
**Confirmed matching `spm_management/`'s implementation, directly against the paper text:**
- **BFS-DFS scheduling is real and matches exactly** (§4.1 "BFS-DFS Schedule"). Nodes are categorized compute-bound / activation-bound / weight-bound; a node queue starts with the graph's source nodes; each committed node's children are enqueued with activation-bound children pushed to the *head* (scheduled as close as possible to their producer, minimizing liveness) and weight-/compute-bound children pushed to the *tail*. Runs in O(nodes+edges) time.
- **The FoM formula matches exactly, coefficients included.** The paper's real Eq. (1):
  `FoM_D = α·(#nodes / UL_D) + β·Σ_{n∈nodes_D}(1/reuse(D,n)) + γ·(Σ_{n∈nodes_D} #ops(n) / Total DNN ops)`
  where `UL_D` is "Unused Liveness" (liveness minus the timesteps D goes unused), `reuse(D,n)` is the max reuse of an element of D within operation n, and the third term is D's operations as a fraction of the whole DNN's. §6 states the exact coefficients used: **α=0.4, β=0.5, γ=0.1** — matching `spm_management/`'s implemented formula term-for-term and constant-for-constant. Tensor **size is not a term in the FoM formula** — it's used separately, as a hard feasibility check during pinning (does the tensor fit in SPM for its entire lifetime), not as a ranking input. (The secondary source's prose paraphrase had blurred "properties tracked per tensor," which does include size/start/end time, with "terms actually inside the FoM formula," which doesn't — that ambiguity is what led the earlier draft of this doc to flag a discrepancy that isn't real.)
- Greedy whole-interval-only pinning (§4.2, Figure 4): tensors are pseudo-sorted by FoM (only comparing tensors whose lifetimes actually overlap — done in O(edges·K) rather than a strict O(edges·log(edges)) sort), then considered highest-FoM-first; a tensor is pinned to SPM only if it has capacity for the *entire* duration it's alive, else it's left in external memory. One extra real detail worth carrying into the port: before checking capacity, the paper first checks for an "overwrite candidate" — an existing SPM occupant whose `EndTS` coincides with the new tensor's `StartTS` and which the new tensor's producing operation consumes — letting the new tensor reuse that freed space directly. This is exactly `onsram_weights.py`'s "Overwrite Optimization," confirming that's the paper-faithful piece to port, not an ad hoc addition.
- Activation-only scope (§3.2 "Pinning Weights") — weights have no reuse across layers, so they're excluded from pinning; matches `spm_management/` and COSMA's convention. (The paper separately studies a variant with weights *pre-loaded* onto a dedicated weight SPM as a point of comparison — Figure 10 — but that's a different architectural configuration, not something the FoM/pinning algorithm itself does.)
- Single shared scratchpad, not multiple (§3.1, Figure 2's architectural template) — matches the planned reuse of `SpmAllocator`'s single unified `memory_budget_bytes` model.
- **Correct headline numbers** (Abstract, Table 1, §7.1, §7.1.4): OnSRAM-Static achieves **1.02–4.8×** inference-latency reduction vs. no SPM management (OnSRAM-Eager: 1.02–3.1×), across 12 real networks (AlexNet, VGG-16, GoogLeNet, Inception-v3/v4, ResNet-50, SSD300, ResNeXt, MobileNetV1, SqueezeNet, PTB-LSTM, Multi-Head Attention) on a 3 TFLOP accelerator, 2MB SPM, 32 GBps external bandwidth, batch size 1. Table 1's GeoMean for an *infinite* SPM (the true upper bound) is 1.76×, with a per-model max of 5.17× (MobileNetV1) — this is the paper's own more precise version of the abstract's rounded "up to 5.2×" framing. **Energy**: average **1.51× (up to 4.1×) for Static**, average **1.23× (up to 2.9×) for Eager** — an earlier draft of this doc mistakenly wrote "1.02–4.8× energy reduction," conflating the latency range with energy; corrected here.

Nothing from `spm_management/`'s documented algorithm is contradicted by the primary text — everything checked out once checked against the right source.

---
run_onsram.py now wires OnSRAM's pinning decisions into real SCALE-Sim via COSMA's unmodified baseline.run_cosma_aware(), compared against a plain baseline.run_baseline() pass, replacing OnSRAM's original closed-form latency estimate entirely. Verified on two real models:
## 5. What Gets Reused From COSMA vs. What's New

COSMA already has a recent precedent for exactly this kind of reuse: the `run_paper_baselines.py` effort (Belady and ILP-greedy replacement policies, neither of them COSMA's own algorithm) already reuses the same generic infrastructure listed below, without ever editing COSMA's own files. We follow the identical pattern.

**Reused as-is, unmodified:**
| From COSMA | What it does | Why it's safe to reuse |
|---|---|---|
| `cosma/helpers/graph_builder.py`'s `load_graph()` | Parses `model.json` into `nodes`/`tensors` dicts | Zero COSMA-specific logic; OnSRAM's own JSON schema is a subset of `model.json`'s |
| `cosma/helpers/model_resolver.py` / `topology_builder.py` | `.tflite` → `model.json` export/caching, SCALE-Sim topology CSV generation | Generic utilities, no ILP/COSMA coupling |
| `cosma/helpers/spm_allocator.py`'s `SpmAllocator` | Verifies/replays a `{(tensor,t): address}` placement against a byte budget | Takes only plain dicts and a budget — no assumptions about *how* the plan was produced |
| `cosma/helpers/baseline.py`'s `run_cosma_aware()` | Drives a real SCALE-Sim simulation pass given a `resident_action`/`spm_plan`/`schedule` | Same generic dict shapes as `SpmAllocator` — this is what gives OnSRAM real, engine-verified cycle counts and DRAM bytes once it produces a plan in this shape, replacing OnSRAM's own closed-form latency formulas entirely |

**Never touched:** `cosma/helpers/cosma_Ilp.py` (COSMA's ILP) and `cosma/run_cosma.py` (COSMA's orchestration) — both under active, separate development. This new work only ever imports their sibling modules' public functions, never these two.

**New, OnSRAM-only code:** everything that implements the FoM scoring, the greedy pinning decision, and the scheduling step — none of that exists anywhere in COSMA, since it's a fundamentally different algorithm.

---

## 6. New File Layout

A new sibling directory to `cosma/`:

```
onsram/
├── docs/
│   └── onsram_integration_plan.md      <- this file
├── onsram_helpers/                     <- OnSRAM-only logic (deliberately NOT
│   │                                      named "helpers/" — avoids shadowing
│   │                                      cosma/helpers/ when both are on
│   │                                      PYTHONPATH at once)
│   ├── fom.py                          <- Figure-of-Merit scoring
│   ├── pinning.py                      <- greedy whole-interval pinning
│   └── scheduling.py                   <- BFS/DFS hybrid sibling scheduling
├── run_onsram.py                       <- single model + budget (mirrors run_cosma.py)
└── run_onsram_experiments.py           <- budget/model sweep (mirrors run_experiments.py,
                                            added once run_onsram.py works)
```

Run with `PYTHONPATH` covering both the repo root and `cosma/` (e.g. `PYTHONPATH=../cosma:..:.` from inside `onsram/`), so `from helpers import graph_builder, spm_allocator, baseline` resolves to COSMA's generic modules unmodified, alongside `from onsram_helpers import fom, pinning, scheduling` for the new logic.

---

## 7. Phased Roadmap

- **Phase A** (this document) — done.
- **Phase B** — done. `onsram_helpers/fom.py`'s `load_layer_meta()` side-loads `model.json`'s raw layer records (shape/params) that `graph_builder.load_graph()` deliberately drops; everything else reuses COSMA's parsed `nodes`/`tensors` dicts unmodified.
- **Phase C** — done (`onsram_helpers/fom.py`, `scheduling.py`, `pinning.py`, `run_onsram.py`). Ported `calculate_fom_corrected()` (paper-faithful reuse-factor table), the BFS-DFS hybrid scheduler (independently hand-verified against `cosma/toy_branching_model.json`'s Kahn's-algorithm walk, string tie-break included), and the greedy whole-interval pinning with the Overwrite Optimization. One real empirical finding surfaced during this port, not present in the earlier draft: the reference's own `get_live_timesteps()` has an exclusive-end range bug that undercounts a tensor's true footprint at its last-use timestep — this is *load-bearing* for the reference's own reported MobileNet@2MB numbers (29-30/31 pinned, 1.53MB peak, §4's 2.03×), which are not physically realizable under a real byte-addressed allocator (confirmed via `cosma/helpers/spm_allocator.py`: two 1.53MB tensors the reference's decision thinks don't overlap actually must coexist at their producer/consumer hand-off, exceeding the 2MB budget). This port uses the paper's own inclusive-lifetime semantics (§4.2) instead, with a narrow, explicit exception so the Overwrite Optimization still functions within COSMA's address model (which has no true buffer-aliasing concept): a reclaim-source tensor vacates one timestep early rather than sharing an address with its replacement. Result: 27/30 tensors pinned on MobileNet@2MB (not 29/30) — lower, but physically valid, independently re-verified by a live `SpmAllocator` replay. See `onsram_helpers/pinning.py`'s module docstring for the full empirical trace.
- **Phase D** — done (`run_onsram_scale_sim()` in `run_onsram.py`). Drives OnSRAM's `resident_action`/`spm_plan`/`schedule` through COSMA's unmodified `baseline.run_cosma_aware()`, compared against a plain `baseline.run_baseline()` pass on the same model — OnSRAM's original closed-form latency estimator is now fully replaced by real, engine-measured cycle counts and DRAM byte traffic. The accounting is a simplified duplicate of `run_cosma.py`'s own idealized-vs-real block (COSMA's spill/retrieve terms drop out entirely, since OnSRAM never spills or retrieves — see §2), reporting DRAM traffic reduction split into ifmap residency credit (from pinning) vs. ofmap residency credit (Eq.3's unconditional on-chip-at-creation assumption, not specific to OnSRAM's choices) so the two aren't conflated. Confirmed on MobileNet@2MB: **71.24% DRAM traffic reduction, 1.024× speedup** versus the no-management baseline, both numbers from real SCALE-Sim simulation, not estimated. Real SCALE-Sim is genuinely slow (~3 minutes for MobileNet's 30 layers, two full passes) — a heartbeat (duplicated from `cosma/run_experiments.py`'s own) prints progress every 20s during it; `--no-scale-sim` skips straight to Phase C's fast decision-only pass.
- **Phase E** — `run_onsram.py` already covers this (`--model`/`--spm-mb` sweeps, `--out-csv`, per-combination logs, `--plot`) — a separate `run_onsram_experiments.py` isn't needed as its own file.
- **Phase F** — validation: sanity-check against `spm_management/output.txt`'s own reported 2.03× (MobileNetV1) and `onsram_bw/OnSRAM_Results.md`'s per-model paper-reference table, and against the real paper's own 1.02–4.8× range from §4. Expect *different* absolute numbers once real SCALE-Sim replaces the analytical model — the target is matching the qualitative pattern, not exact byte-for-byte numbers, same standard COSMA's own paper comparison already holds itself to.
- **Phase G** (later, explicitly out of scope for this first pass) — the weight/activation SPM-congestion breakdown + chart, and the fused/unfused bandwidth sensitivity study from `onsram_bw/`.
