# Baseline Construction: TFLite-linear × {Default,MPMF} × {Belady,ILP-greedy}

Standing reference for the paper's own comparison baselines (arXiv:2311.18246,
§V-A2) — what's built, what's verified, and what's left. Update this file as
work lands; it's meant to stay current, not be a one-time snapshot (same
convention as `results_plan.md`).

## 1. Purpose

`results_plan.md`'s mapping table has always flagged this as "open future
work, substantial reimplementation effort, not started": without these 4
baselines, our % non-compulsory-byte reduction numbers aren't the same
measurement as the paper's own 84%/85% headline figures — we've been
comparing COSMA against our own weaker baseline (plain SCALE-Sim, no
cross-layer sharing), not against what the paper itself compares against.
This effort closes that gap.

**Hard constraint**: this must run in parallel with ongoing work on the main
COSMA pipeline without breaking it. Every deliverable below is a new file;
existing files are only ever imported from (their already-public,
already-multi-caller functions), never edited. See §3.

## 2. Paper's own definitions (§V-A2, confirmed by reading the actual PDF)

- **Operator Scheduling**: "(i) The default operator scheduling from the
  PyTorch implementation... and (ii) the operator schedule with minimum peak
  memory footprint (MPMF)." — the MPMF schedule is explicitly the §III-E1/
  Eq.13-15 schedule, i.e. exactly what `cosma_Ilp.compute_true_mpmf_bytes()`
  already computes and returns.
- **Memory Allocation**: "a linear allocation scheme used in TensorFlow
  Lite... designed for memory allocation for hardware accelerators" — the
  paper cites TFLite's real `simple_memory_arena.cc`.
- **Tensor Replacement**: "(i) Belady's algorithm, which always replaces the
  tensor to be accessed in the furthest future, and (ii) an ILP-based greedy
  algorithm which provides a locally optimal decision that generates the
  least off-chip data accesses each time replacement is needed."
- **Reported finding**: Belady is *not* optimal — it generates *more*
  off-chip accesses than the greedy-ILP policy under the same
  schedule/allocator. Reproduced by this implementation — see §7.

4 combinations built: Default+Belady, Default+ILP-greedy, MPMF+Belady,
MPMF+ILP-greedy — plus an optional 5th "cosma_native" row (COSMA's own real
pipeline, unmodified) for direct side-by-side comparison.

## 3. Design: why WHAT/WHEN vs. WHERE is forced, not chosen

Fetched and read TFLite's actual `simple_memory_arena.cc`/`arena_planner.cc`
(github.com/tensorflow/tensorflow, commit `188ddbad6557313713d7de5940daea6b22ae6b7`,
checked 2026-09-10). Finding: `ArenaPlanner`/`SimpleMemoryArena` have **no
capacity constraint and no eviction concept anywhere** — the arena just grows
to fit whatever lifetime intervals it's given. This means "TFLite's linear
allocator" cannot, by itself, produce a spill/retrieve decision. The paper's
2×2 factorization (schedule × replacement, both independent of a placement
scheme held fixed across all 4 cells) already presupposes exactly the split
used here: a **replacement policy** decides *which tensor's residency
episode exists when* → produces `resident_action: {(tensor_id,t):
'C'|'P'|'S'|'R'}`; the **linear allocator** decides *where each episode sits
in address space* given that fixed set of episodes → produces `spm_plan:
{(tensor_id,t): address}`. This is exactly COSMA's own existing vocabulary
(`helpers/spm_allocator.py`), so the result feeds unmodified into
`helpers/baseline.py`'s existing `run_cosma_aware()` — the same trusted
accounting path already used for every COSMA number on record.

Audited every branch of `SpmAllocator`/`run_cosma_aware`/`_run_layers` for a
hidden COSMA-ILP-specific assumption: found none. COSMA's own Eq.8 (at most
one spill per tensor) is a modeling choice of COSMA's ILP, not a
`SpmAllocator` requirement — the replacement policies here are free to
spill/retrieve a tensor multiple times, which a real replacement policy
legitimately should do (confirmed this doesn't break anything — see §7).

**One deliberate exception**: `run_cosma.py`'s idealized-vs-real byte
accounting (~15 lines, `total_non_compulsory_access_bytes` etc.) is inlined
in `run_cosma()`'s body, not a reusable function. `run_paper_baselines.py`'s
`_account()` is a disclosed **duplicate** of that block, not an import —
extracting a shared helper would mean editing `run_cosma.py`, which this
effort must not touch.

## 4. Mapping/status table

| File | Status | Notes |
|---|---|---|
| `helpers/tflite_arena_allocator.py` | **Done, verified** | Real TFLite algorithm (best-fit-by-gap, whole-horizon + size-descending order). See §5, §7. |
| `helpers/schedule_variants.py` | **Done, verified** | `default_operator_schedule()`, `mpmf_operator_schedule()` (inverts `compute_true_mpmf_bytes()`'s existing return, zero `cosma_Ilp.py` changes needed). See §7. |
| `helpers/replacement_engine.py` | **Done, verified** | Shared WHAT/WHEN loop. A real layer-id-vs-timestep-position bug was found and fixed during development — see §6/§7. |
| `helpers/belady_policy.py` | **Done, verified** | Furthest-next-use, full lookahead. |
| `helpers/ilp_greedy_policy.py` | **Done, verified** | Local per-decision ILP, minimize evicted bytes. Reproduces the paper's own Belady-suboptimal finding — see §7. |
| `run_paper_baselines.py` | **Done, verified end-to-end** | All 4 combos + `cosma_native` row, on a real model through real SCALE-Sim. See §7. |
| Activation+parameter tracking | **Not in scope here** | Separate, already-tracked gap (`results_plan.md` §6 item 2) — these baselines are activation-only, same as the rest of this codebase. |
| ImageNet-scale models (Inception-V3/ResNet-50/DenseNet-121) | **Not yet attempted** | Only tested on the small custom DenseNet fixture (`_exported/fake/`) so far. MPMF schedule ILP + per-decision ILP-greedy solves may both be slow at this scale — untested. |

## 5. Algorithm — TFLite linear allocator

Ported (not imported) from `tensorflow/lite/arena_planner.cc`'s
`CreateTensorAllocationVector()` (placement order) and
`tensorflow/lite/simple_memory_arena.cc`'s `Allocate()` (placement itself),
commit `188ddbad6557313713d7de5940daea6b22ae6b7`, checked 2026-09-10:

1. Group `resident_action` into episodes (maximal contiguous C/P/R runs per
   tensor).
2. Sort: episodes spanning the whole schedule horizon first (by tensor id);
   everyone else by size descending, ties broken by `(t_start, tensor_id)`.
3. Place each episode via **best-fit-by-gap** against only its
   time-overlapping already-placed neighbors (non-overlapping episodes
   freely reuse the same address) — deliberately different from
   `spm_allocator.compact_spm_plan()`'s first-fit heuristic, which serves an
   unrelated visualization purpose.

Disclosed simplifications vs. real TFLite: no byte alignment, no in-place/
aliasing buffer sharing — both consistent with what the rest of this
codebase already does/doesn't model.

## 6. Algorithm — Belady / ILP-greedy

**Belady**: rank resident, evictable candidates by furthest next-access
position (never-reused-again ranks first, i.e. infinite distance), take
greedily until the deficit is covered. Full lookahead over the whole
remaining fixed schedule — legitimate, since Belady/MIN is *defined* as the
offline-oracle algorithm and the schedule is already fixed before
replacement runs.

**ILP-greedy**: per eviction decision, `x[a] ∈ {0,1}` per candidate,
`minimize Σ x[a]·size(a)` subject to `Σ x[a]·size(a) ≥ deficit_bytes`. Every
candidate is guaranteed a real future consumer (dead tensors are freed for
free, before this is ever called), so evicting it now deterministically
costs one S + one eventual R = `2·size(a)` — the same multiplier for every
candidate, so minimizing evicted bytes is order-equivalent to minimizing
true eventual off-chip cost, while staying genuinely *local* (no lookahead
into whether an evicted tensor gets evicted again before use — that's
COSMA's own joint-ILP territory).

**Correctness note, found during development**: `tensors[a].consumer_layers`
stores *layer ids*, but `replacement_engine.py` walks the schedule by
abstract *timestep t*. Under a genuinely reordered schedule (confirmed to
happen for real — see §7), layer id and timestep position are different
numbers. The engine precomputes `position_of_layer`/`consumer_positions`
once and passes the latter to both policies, rather than letting each policy
compare `consumer_layers` against `t` directly (which would have been
silently wrong under any reordered schedule).

## 7. Verification log

**Phase 1 — `tflite_arena_allocator.py`** (`python3 -m helpers.tflite_arena_allocator`):
Case A — hand-traced step-by-step against a `toy_spill_model.json`-shaped
`resident_action`; output matched the hand derivation exactly (tensors 10/11
share address reuse correctly across non-overlapping episodes). Case B —
3×100B tensors mutually resident at once, budget=250: correctly raised
`TfliteArenaAllocationError`. (A more subtle pure-fragmentation, not-just-
overcommit, adversarial case was attempted by hand for this specific
best-fit-by-gap algorithm during planning and repeatedly resolved cleanly —
best-fit is a meaningfully stronger heuristic than `compact_spm_plan()`'s
first-fit; a genuine real-world fragmentation failure showed up unprompted
in Phase 5 instead, see below — more convincing than a constructed one.)

**Phase 2 — `schedule_variants.py`** (`python3 -m helpers.schedule_variants`):
`toy_spill_model.json` — live-solved MPMF schedule is `[(0,1),(1,2),(2,0),(3,3)]`
(200 bytes = M_R exactly), a **genuine reorder** vs. default
`[(0,0),(1,1),(2,2),(3,3)]` (210 bytes). This corrected an initial planning-
time assumption that this fixture's MPMF schedule would match default — it
doesn't; re-derived by hand and confirmed by a live solve before writing the
regression assertion. `_exported/fake/model.json` — MPMF schedule confirmed
byte-identical to default (622592 bytes both).

**Phase 3 — `replacement_engine.py` + `belady_policy.py`**
(`python3 -m helpers.replacement_engine`, `python3 -m helpers.belady_policy`):
`toy_spill_model.json` @ budget=200 reproduces the exact `resident_action`
COSMA's own optimal ILP solve finds (S/R = 10 bytes each, on tensor 10) —
hand-traced through the engine's own step order (this caught and fixed a
transcription slip from an earlier, less careful trace: tensor 11 needs an
explicit `P` at t=2, not just its `C` at t=1). Budget=110 (< M_R=200)
correctly raises `ReplacementInfeasible`.

**Phase 4 — `ilp_greedy_policy.py`** (`python3 -m helpers.ilp_greedy_policy`):
Purpose-built divergence fixture (X=150KB far-future, Y=Z=60KB near-future,
deficit=100KB): Belady evicts `{X}` (150KB, 300KB eventual traffic);
ILP-greedy evicts `{Y,Z}` (120KB, 240KB eventual traffic) — strictly less,
**reproducing the paper's own reported Belady-suboptimal finding**
mechanically, not by coincidence. Cross-check on `toy_spill_model.json`
(only one evictable candidate ever exists there): both policies agree
exactly, as they must.

**Phase 5 — wired into real `baseline.run_cosma_aware()`**
(`_exported/fake/model.json`, M_R=524288, MPMF-proxy=622592):
- @550KB: all 4 new baselines **and** `cosma_native` produce **identical**
  `total_non_compulsory_access_bytes` (425984), zero `SpmAllocationError`.
  COSMA's own ILP finds nothing better than TFLite-linear+Belady/greedy on
  this small model at this budget — expected, consistent with this
  project's earlier finding that `free_schedule` doesn't improve this
  particular fixture either.
- @525KB: **`default+belady` (and all 4 combos) genuinely fail** —
  `TfliteArenaAllocationError`: a 160KB tensor's episode can't fit a gap
  between two other placed regions (196608B + 163840B occupied, 537600B
  budget) even though enough total free space nominally exists. This is the
  **same class of real fragmentation failure already documented in
  `spm_allocator.py`'s `compact_spm_plan()` docstring** (found independently
  there on this same model/architecture) — a genuine, unprompted, real-model
  demonstration of exactly the TFLite-weakness-vs-COSMA-joint-ILP gap the
  paper's comparison exists to reveal, not a constructed example.
- @530KB, @540KB: all combos succeed again, matching `cosma_native` exactly
  (425984 bytes).

**Phase 6 — `run_paper_baselines.py`**: full CLI run on `_exported/fake/model.json`
across multiple budgets, `--out-csv`, `--no-cosma-native` all exercised;
output matches the Phase 5 findings above exactly (same driver, same
underlying calls).

## 8. What blocks full comparability, ranked

1. **ImageNet-scale models untested** — only the small custom DenseNet
   fixture has been run end-to-end. The MPMF schedule ILP and the per-
   decision ILP-greedy solve are both real ILP solves (see
   `cosma_Ilp.compute_true_mpmf_bytes()`'s and `ilp_greedy_policy.py`'s own
   CBC-timeout-mislabeling caveats) — untested at DenseNet-121/Inception-V3
   scale, where this project has already seen CBC struggle on other ILPs.
2. **Activation+parameter tracking** — these baselines inherit the same
   activation-only scope as the rest of this codebase (`results_plan.md`
   §6 item 2, not re-litigated here).
3. **TFLite alignment/in-place aliasing** — disclosed, low-impact
   simplifications (§5), consistent with what the rest of this codebase
   already does/doesn't model, not expected to matter for the comparison's
   own validity.

## 9. Recommended next test to run

Run `run_paper_baselines.py` on a model with genuine parallel branches (a
skip-connection-bearing real model, not just the small linear-ish DenseNet
fixture) across a budget sweep spanning `M_R` to `MPMF` — the 525KB
fragmentation failure in §7 suggests tighter budgets are where this
comparison actually gets interesting; a model with more concurrent tensors
alive at once should produce a real, non-degenerate gap between COSMA-native
and the 4 baselines, not just the ties seen so far on the small fixture.
