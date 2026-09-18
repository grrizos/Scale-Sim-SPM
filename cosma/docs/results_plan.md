# Results Plan: mapping our evaluation onto the COSMA paper's

Standing reference for running tests that produce genuinely comparable
numbers to the COSMA paper (arXiv:2311.18246), and for knowing exactly which
numbers *aren't* comparable yet, and why. Update this file as gaps close —
it's meant to stay current, not be a one-time snapshot.

## 1. Purpose

Every individual gap between our implementation and the paper's own
evaluation has been noted somewhere in `STATUS.md`/`ITERATION_HISTORY.md`
over time, but never assembled into one place describing *how to actually
run a test* that means the same thing as one of the paper's own numbers.
That's what this file is for.

## 2. Paper's evaluation methodology (§V-A, arXiv:2311.18246)

**Three memory budgets per model** (§V-A2):
- **`M_R`** ("Minimum Memory Required"): *"the maximum memory required by a
  single operator in the DNN. This is the sum of the sizes of all input and
  output tensors that must reside in memory during the operation."*
- **`M_P`** ("Minimum Peak Memory Footprint... required for executing the
  entire DNN"): the true, schedule-optimal peak — computed via COSMA's own
  §III-E1 optimization (Eq.13–15) for human-designed DNNs, or via a separate
  tool called HMCOS for NAS models.
- **`M_H`** ("Hybrid"): `(M_R + M_P) / 2`.

**Primary metric**: *"non-compulsory off-chip data access volume"* — bytes
spilled plus bytes retrieved. Secondary metric: solving time (seconds).

**Two tensor-tracking settings**: "activation tensors only" vs. "activation
tensors and parameter tensors." Parameter tensors are described as *"used
only for single operators, e.g., the weight tensors of a convolution
layer,"* while activation tensors are *"used across different operators."*

**Comparison baselines** (§V-A2), 4 combinations: TensorFlow-Lite's linear
allocator × {default schedule, MPMF schedule} × {Belady's algorithm,
ILP-based greedy replacement}.

**Solver/hardware** (§V-A): Gurobi, Apple M1 Pro, 16GB RAM, 24-hour time
limit per ILP call. Human-designed DNNs solved in **0.296s on average**; NAS
models (via heuristics) in ~2 minutes.

**Models tested** (§V-A1):
- 10 human-designed (Figure 3): image classification — **ResNet-50,
  DenseNet, ResNeXt**; video classification — R2Plus1D, S3D; semantic
  segmentation — FCN, L-RASPP, DeepLabV3; transformer-based — Transformer,
  ViT.
- 4 NAS-generated (Figure 4, Table II): PNASNet-5, AmoebaNet-D, NASNet-A,
  DARTS — noted as having *"complex graph structure and irregular wiring
  between nodes,"* requiring divide-and-conquer/fixed-schedule heuristics
  rather than direct solving.

## 3. Mapping table: paper concept → our implementation status

| Paper concept | Our status | Where |
|---|---|---|
| `M_R` | **Have it, faithful** | `cosma_Ilp.compute_structural_minimum_bytes()` — max over operators of activation-input + output bytes. Now confirmed against the primary source (§V-A, Fig.3 caption): the paper's own `M_R` is *also* activation-only — ours isn't under-counting it. The paper separately reports a parameter-inclusive `M_Rp` (same definition, bigger tensor set) alongside `M_R` for every human-designed DNN; this port doesn't compute that second number yet (see the activation+parameter row below). |
| `M_P` | **Have it, real §III-E1/Eq.13-15 solve** | `cosma_Ilp.compute_true_mpmf_bytes()` — a genuinely separate, free-schedule ILP (`C[a,t]` a real variable, not fixed), reusing none of `build_cosma_model()`'s memory-allocation machinery (no `L`/Eq.9/10/11 — the paper's own text: "memory allocation is not considered" in this mode). `compute_mpmf_bytes()` (the old fixed-schedule proxy) is unchanged and still used by `--bounds-only`'s fast default path — `M_R ≤ true_M_P ≤ MPMF-proxy` always holds, verified on 6 models (see §4). Exposed via `visualize_spm.py --bounds-only --true-mpmf` (opt-in — a real solve, not instant). |
| `M_H` | **Have it** | `(M_R + true_M_P) / 2`, computed inline (`visualize_spm.print_true_mpmf()`) wherever `--true-mpmf` is used — no dedicated function needed, it's one line. |
| Primary metric (spill+retrieve bytes) | **Have it now** | `run_cosma.py`'s `total_non_compulsory_access_bytes` (added alongside this doc) = `total_idealized_spill_bytes + total_idealized_retrieve_bytes + total_real_retrieve_bytes`. |
| Activation-only tracking | **Have it — it's the only mode** | `graph_builder.load_graph()`, unconditional. |
| Activation+parameter tracking | **Not implemented** | Corrected against the primary source (§V-B): the paper's own "both activation tensors and parameter tensors" setting is *not* a different formulation — same Eq.1-12, same C/P/S/R model, just weights included in tensor set `A`. It's one of the paper's two standard settings, reported for all 10 human-designed DNNs in Fig.3 (as `M_Rp`/`M_Hp`/`M_Pp`), not a side study. The ILP side of this port needs no new constraint logic to match it — just extending `graph_builder.load_graph()`'s tensor scope. See §6 below for the real remaining question, which is about the *SCALE-Sim-simulated* numbers, not the ILP. |
| Comparison baselines (TFLite × Belady/greedy) | **Not implemented — open future work** | Our baseline (SCALE-Sim's own default per-layer buffers, no cross-layer sharing) is a different, weaker comparison point. |
| Operator scheduling | **Implemented in the main pipeline too** | `build_cosma_model(..., free_schedule=True)` — `C[a,t]` is now a real decision inside the *main* spill/retrieve pipeline itself (not just the isolated `M_P` model), and the ILP's chosen order drives a real SCALE-Sim re-simulation via `baseline.run_cosma_aware()`'s `schedule` param. Opt-in, default `False` (byte-identical to every previously published number). Verified on both toy fixtures, the small custom DenseNet fixture (real SCALE-Sim run), and ResNet-20-CIFAR10 — see §4/§7. Not yet run at ImageNet scale (Inception-V3/ResNet-50/DenseNet-121) — see §6 item 1. |
| Divide-and-conquer (NAS-scale) | **Permanently out of scope** | Per explicit standing project direction. |
| Solver | PuLP/CBC, not Gurobi | Confirmed meaningfully slower: 178s on Inception-V3 vs. paper's 0.296s average. Acknowledged, fixed difference. |

## 4. Model roster and bounds

| Model | `M_R` | MPMF proxy | **True `M_P`** | `M_H` | Full SCALE-Sim run done? |
|---|---|---|---|---|---|
| `toy_spill_model.json` | 200 B | 210 B | **200 B (= M_R)** | 200 B | N/A — no real conv params |
| `toy_branching_model.json` | 144.00 KB | 208.00 KB | **144.00 KB (= M_R)** | 144.00 KB | N/A — no real conv params |
| MobileNetV2-CIFAR10 | 60.00 KB | 60.00 KB | 60.00 KB (squeeze-forced equal) | 60.00 KB | Yes — 64KB: 32.5% DRAM reduction, 1.0016× speedup |
| ResNet-20-CIFAR10 | 192.00 KB | 192.00 KB | 192.00 KB (squeeze-forced equal) | 192.00 KB | Yes — 256KB: 91.1%, 1.0299× |
| SqueezeNet-small-CIFAR100 | 80.00 KB | 80.00 KB | 80.00 KB (squeeze-forced equal) | 80.00 KB | Yes — 96KB: 92.0%, 1.3599× |
| Inception-V3 | 8103.38 KB | 8103.38 KB | not yet run — expect a real solve-time jump (§6) | — | No — never run to completion at a real budget |
| ResNet-50 | 9408.00 KB | 9408.00 KB | not yet run | — | Partial — 76.21% reduction, 1.0355× speedup at "80MB" (unit ambiguous in the source note — likely not a budget in the same KB units used elsewhere; flagged, not silently normalized) |
| DenseNet-121 (full ImageNet-scale) | 6328.25 KB | 8232.00 KB — **first real `M_R != MPMF-proxy` gap** | not yet run — the main spill/retrieve ILP alone already ran 47+ min of CBC time unbounded on `wil` before being killed (item 29); the scheduling ILP is a separate, likely also-slow solve | — | No — hit `Not Solved` before the time-limit fix; not yet re-run to completion either way |
| Small custom DenseNet (18 conv layers, 2 blocks × 4 units, growth rate 12, 32×32 input — this session's own fixture, random weights) | 512.00 KB | 608.00 KB | **608.00 KB (no improvement over the fixed schedule — free scheduling confirmed today's arbitrary op order was already optimal for peak footprint here)** | 560.00 KB | Yes — 550KB: 84.2% DRAM reduction, **0.9985× speedup (net slower)** — genuine spill/retrieve tradeoff cost, first real (non-toy) demonstration of a retrieve becoming a *new* bottleneck |

**A real, informative split emerged**: on every model where the fixed-
schedule `MPMF` proxy already equalled `M_R` (all 5 originally-tested real
models, plus both toy fixtures), `true_M_P` is *mathematically forced* to
equal `M_R` too (by `M_R ≤ true_M_P ≤ MPMF-proxy`) — confirmed by an actual
solve, not just the inequality, on every one of them. But the two toy
fixtures (the only models with a real `M_R != MPMF-proxy` gap that have
actually been solved) split differently: `toy_spill_model.json`/
`toy_branching_model.json` both improved all the way down to `true_M_P ==
M_R` (free scheduling found a real, better order), while the small custom
DenseNet fixture did **not** improve at all (`true_M_P == MPMF-proxy` — the
existing fixed order was already schedule-optimal for peak footprint, even
though it isn't for spill/retrieve minimization, which is a genuinely
different objective). Whether DenseNet-121 itself would improve is unknown
— not yet run (see above).

**Main-pipeline free scheduling** (`free_schedule=True`, distinct from the
`true_M_P` column above, which only ever optimizes peak footprint): on
ResNet-20-CIFAR10, an actual `free_schedule=True` solve (real SCALE-Sim
re-run, 240s CBC limit) reproduced the fixed schedule's numbers exactly (0
non-compulsory bytes, 78.3% reduction, 1.0000x, both) — no improvement, same
pattern as `true_M_P` on this architecture class. On the small custom
DenseNet fixture, free scheduling found *no* reduction in total
non-compulsory bytes either (425984 both) — but did find a real, different
schedule that avoids landing a retrieve on layer 16's `CONCAT` (free in
baseline, a new bottleneck under the fixed COSMA schedule), improving real
simulated cycles 61458 → 58130 at the same budget. This is a genuinely new
kind of result: scheduling freedom can matter for real performance even
when it doesn't move the paper's own primary metric at all. Both real runs
had 0 `SpmAllocator` violations. A deliberately adversarial branching toy
fixture (`toy_branching_model.json`) confirmed the expected complexity
ceiling: the ASAP/ALAP pair-filter prunes 0% of pairs there (vs. ~92% for
the DenseNet fixture), and the solve didn't reach `Optimal` within ~9
minutes of continuous CBC time. See `ITERATION_HISTORY.md` item 32 for the
full detail.

**Overlap with the paper's own model list**: `ResNet-50` and `DenseNet` are
both literally in the paper's own 10 human-designed models (§V-A1) — not
approximations. Our small custom DenseNet is a fast-solving *stand-in* for
the same architectural family, not a literal replication of the paper's own
(unspecified-size) DenseNet run. We have no analog for the paper's 4
NAS-generated models (PNASNet-5, AmoebaNet-D, NASNet-A, DARTS) — no
divide-and-conquer heuristic exists here to make those tractable, per the
standing scope decision.

## 5. What we can run today, faithfully

For any model:
```bash
# 1. Get M_R, the fast MPMF proxy, AND the real M_P/M_H (real ILP solve,
#    not instant -- can be slow on a large model, see §6/§4's DenseNet-121
#    note)
PYTHONPATH=..:. python3 visualize_spm.py --model-json <model> \
    --bounds-only --true-mpmf

# 2. Run at M_R, true M_P, M_H (and the old fixed-schedule proxy too, if it
#    differs from true M_P -- worth comparing directly when it does)
PYTHONPATH=..:. python3 run_experiments.py --models <model> \
    --budgets-kb <M_R> <M_H> <true_M_P> --config <config>
```
Report `total_non_compulsory_access_bytes` (paper-comparable primary metric)
alongside our own `dram_traffic_reduction_pct`/`speedup` (broader, real-
SCALE-Sim-simulated, not the same measurement as the paper's numbers, but
genuinely useful in its own right). All three of `M_R`/`M_H`/`M_P` are now
real, reportable numbers — no more caveats needed on that front. If
`--true-mpmf` is skipped (e.g. for speed on a large model), say explicitly
that only the fixed-schedule proxy was used, not real `M_P`.

## 6. What blocks full comparability, ranked

1. **`M_P`/`M_H` at production scale** — implemented and verified correct
   (§4), but Inception-V3/ResNet-50/DenseNet-121 haven't been run through it
   yet, since `C[a,t]` becoming a full `|T|x|A|` binary block is exactly the
   paper's own `O(|T|x|A|^2)` worst case (§III-F) — expect the same kind of
   solve-time jump already seen for the main pipeline on DenseNet-121 (item
   29: 47+ min of CBC time, killed before finishing).
2. **Activation+parameter tracking** — corrected against the primary source
   (see the mapping table above): the paper's own ILP needs no special
   handling for weights, it just includes them in `A`. That means this
   item isn't really an ILP-design question at all, it's a
   *SCALE-Sim-simulation-accounting* question, specific to this port: an
   activation's `'C'` (create) event maps naturally onto a real, already-
   free SCALE-Sim event (computing a layer's own ofmap has no DRAM cost
   in the engine, full stop) — but a weight tensor's `'C'` has no
   equivalent free event to map onto, since a weight is never "computed
   on-chip," it always needs at least one real DRAM fetch. The paper's own
   ILP sidesteps this cleanly by never charging *any* `'C'` event in its
   objective (Eq.12 only ever sums `S`/`R` terms) — compulsory cost, for
   any tensor, activation or weight, is defined as outside the optimized
   quantity entirely, so the ILP itself never needs to know weight-`'C'`
   is "different." Simulating that same plan for real, though, requires
   this port to decide how `baseline.run_cosma_aware()` charges a weight's
   first, unavoidable fetch — the paper's own analytical accounting never
   has to answer that, since it has no engine underneath it at all. Worked
   out carefully before trusting any resulting numbers.
3. **Comparison baselines** (TFLite linear allocator × Belady/greedy) — open
   future work, substantial reimplementation effort, not started.
4. **Full operator rescheduling for the main (spill/retrieve) pipeline** —
   now implemented (`build_cosma_model(..., free_schedule=True)`, opt-in,
   default off), and re-simulated for real via `baseline.run_cosma_aware()`'s
   `schedule` param — not just an isolated bound anymore. What's left:
   only exercised so far on both toy fixtures, the small custom DenseNet
   fixture, and ResNet-20-CIFAR10 (see §4/§7) — Inception-V3/ResNet-50/
   DenseNet-121 haven't been attempted under `free_schedule=True` yet, and
   a deliberately adversarial branching toy fixture showed this can hit
   real, substantial solve-time cost when the ASAP/ALAP pair-filter can't
   prune much (0% pruning on that fixture vs. ~92% on the DenseNet one).
   Divide-and-conquer for NAS-scale graphs remains out of scope, per
   standing project direction.
5. **Solver/hardware** (PuLP/CBC vs. Gurobi, different machine) — acknowledged
   fixed difference, not something to chase.

## 7. Recommended next test to run

DenseNet-121 (full ImageNet-scale) is the one real (non-toy, non-tiny) model
with a genuine `M_R != MPMF` gap that's never been run to completion — it hit
`Not Solved` before the time-limit fix landed. Now that solves are unbounded,
re-running it at a budget inside `(6328.25, 8232.00)` KB would be the first
production-scale (not toy-fixture-scale) real spill/retrieve demonstration.
Worth doing once ready to let a potentially long solve run (the earlier
attempt burned 47+ minutes of CBC CPU time before being killed — see
`ITERATION_HISTORY.md` item 29's discussion of why this is expected for a
121-layer, densely-connected model near its spill/retrieve boundary).

With `free_schedule=True` now implemented (item 32), DenseNet-121 is also
the natural next candidate for it specifically *because* it's the one real
model with a genuine `M_R != MPMF` gap — every model where `free_schedule`
has been tried so far either had no gap to begin with (ResNet-20) or didn't
improve on it (the small DenseNet fixture). Expect `free_schedule=False` to
already be slow here (above); `True` adds the full `|T|x|A|` `C` block on
top, so budget for it being substantially slower still — start with
`free_schedule=False` and only attempt `True` once that completes.
