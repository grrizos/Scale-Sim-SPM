# Continue Here — 2026-09-16 session

Picking this back up: read this file first, then the "Next steps" section at
the bottom tells you exactly what to do. Branch: `OnSram`.

## What happened this session, in order

1. **`spm_common/` refactor** — already committed (by you, on `OnSram`,
   commit `3ff38c1` "small changes") before this log was written, so it's
   safe and not part of the uncommitted work below. `graph_builder.py`,
   `model_resolver.py`, `spm_allocator.py` moved out of `cosma/helpers/`
   into a new repo-root-sibling `spm_common/`, since OnSRAM was importing
   them unmodified anyway (COSMA's `topology_builder.py`/`baseline.py`
   stayed put — those *are* deliberately duplicated per-paper for
   simulation isolation, unlike the three that moved).
2. **Model rosters built** for all three papers, checked against actual
   `cosma/_exported/` availability, same format as the pre-existing
   `cosma/docs/paper_model_roster.md`:
   - `onsram/docs/onsram_model_roster.md` (new) — 5/12 ready today.
   - `smm/docs/smm_model_roster.md` (new) — 2/6 ready today, 1 more
     likely-available-but-unconfirmed-variant.
3. **Third paper identified**: Zouzoula, Maleki, Azhar, Trancoso.
   "Scratchpad Memory Management for Deep Learning Accelerators." ICPP '24.
   doi.org/10.1145/3673038.3673115. User has the PDF. Its port
   (`smm_policy_selector.py`, `smm_scalesim_runner.py`) currently lives as
   loose top-level files on the **`sim-opt`** branch — NOT merged into
   `OnSram`'s `cosma/`+`onsram/`+`spm_common/` structure yet. `smm/docs/`
   on this branch only has the roster so far, no ported code.
4. **Found and confirmed a real simulation-fidelity gap** (see next
   section) — added a diagnostic check for it, ran it against real
   infrastructure, it fired for real on both COSMA and OnSRAM.
5. **Checked paper fidelity against primary sources** (user supplied the
   COSMA and SMM PDFs directly) — this reversed one of my own earlier
   claims. See "Paper-fidelity findings" below.
6. **Agreed a governing principle** for what counts as an in-scope fix
   vs. scope creep — see that section, it resolves an open question from
   earlier in the session.
7. Session ended on **one unresolved technical question** blocking the
   next implementation step — see "Open question" below.

## The core finding: budget vs. working-set gap

`SpmAllocator` verifies resident *activation* tensors fit the budget
against each other. Separately, `_make_memory_system()` (in
`cosma/helpers/baseline.py` and `onsram/onsram_helpers/scale_sim_runner.py`)
sizes each layer's own SCALE-Sim buffers to that layer's real tensor bytes,
completely unaware of the residency budget or of `SpmAllocator`'s state.
Neither ever compares notes — so a plan `SpmAllocator` reports as "0
violations" can still, once you add the currently-executing layer's own
filter/weight fetch on top, genuinely exceed the stated budget.

Added a conservative diagnostic check for this (filter bytes are always
safe to add — COSMA/OnSRAM never track them, so no double-counting risk;
ifmap/ofmap deliberately excluded, genuine double-counting ambiguity with
residency, still unresolved) in both:
- `cosma/helpers/baseline.py`'s `_run_layers()` (new `budget_overflow_events`
  block + `[COSMA SPM] WARNING: ...` print)
- `onsram/onsram_helpers/scale_sim_runner.py`'s equivalent loop (identical
  pattern, `[OnSRAM SPM] WARNING: ...`)

**Ran it for real, it fired both times:**
- COSMA, ResNet-20-CIFAR10 @ 200KB (near its own `M_R` ≈ 192KB, tight
  budget): **4 of 32 timesteps** over budget, worst case **8,448 bytes**
  over at t=21.
- OnSRAM, MobileNet @ 2MB (the project's own default fixture): **3 of 30
  timesteps** over budget, worst case **2,502,656 bytes** over at t=26 —
  verified this wasn't a bug in the check itself: layer 26's own weights
  alone are 4,198,400 bytes (a 1024×1024 1×1 conv at float32), more than
  double the entire 2MB SPM budget by itself.

This is real, not hypothetical — it fired on infrastructure that had
already been run and trusted (this exact ResNet-20 run underlies numbers
in `results_plan.md`).

## Paper-fidelity findings (from primary sources)

User supplied the actual COSMA and SMM PDFs mid-session. Findings from
those changed what I'd said earlier, based only on this project's own
secondary notes:

- **COSMA (arXiv:2311.18246) — CONFIRMED, reverses my earlier claim.**
  §V-B: the paper's own main evaluation runs **two** settings —
  "(i) only the activation tensors... and (ii) both activation tensors and
  parameter tensors" — same ILP (Eq. 1-12), just a bigger tensor set `A`
  in setting (ii). Not a side study: Fig. 3 reports `M_Rp`/`M_Hp`/`M_Pp`
  (parameter-inclusive budgets) alongside `M_R`/`M_H`/`M_P` for **all 10**
  human-designed DNNs, same ~0.3s solve time for both settings. This
  port's `cosma/docs/results_plan.md` previously said weight-tracking
  needed "a virtual pre-schedule spill/retrieve seeding, or an equivalent
  constraint change" because "a weight's first appearance is never free" —
  **already corrected in `results_plan.md` this session** (both the
  mapping-table row and §6 item 2) — the paper's ILP needs no such thing;
  `Eq.12` never charges any `'C'` event at all, weight or activation, so
  the asymmetry only exists in *this port's SCALE-Sim-simulation-mapping
  layer*, not the ILP itself.
- **SMM (Zouzoula et al., ICPP '24) — CONFIRMED.** §3.1, the paper's own
  main "Optimization problem formulation": `GLB ≥ I_Tile + F_Tile + O_Tile`
  (Eq. 1) and the double-buffered version (Eq. 2) both require filter tile
  space in the *same* budget as ifmap/ofmap. Weights compete for budget
  by design, in the core algorithm, not an extra.
- **OnSRAM — NOT yet confirmed from a primary source.** Only have this
  project's own `onsram_integration_plan.md` notes: §3.2 "Pinning Weights"
  excludes weights from pinning on a *reuse* argument (no cross-layer
  reuse benefit), and describes a dedicated-weight-SPM variant (paper's
  own Fig. 10) as "a different architectural configuration" — suggestive
  that the paper's *main* algorithm may not require weights to share a
  budget with activations at all, but this is inference from secondary
  notes, not confirmed the way COSMA/SMM now are. **If you have the OnSRAM
  PDF, that's the single highest-value thing to hand over next** — it
  directly resolves whether OnSRAM's own filter-overflow finding above
  needs fixing or should be preserved as a faithful reproduction of the
  paper's own scope.

## Governing principle (agreed this session)

> We only change an algorithm's own modeling scope if the *paper's own
> algorithm* does that thing. If a paper doesn't model something (e.g.
> possibly OnSRAM not budgeting for weights), that's a faithful limitation
> to preserve, not a bug to fix — the whole point is a fair comparison of
> what each paper actually claims, not our own idea of what's "more
> correct."

Applying it with what's confirmed so far:
- **COSMA: green-lit** to add activation+parameter tracking — paper
  confirms this is real, in-scope, core-algorithm behavior.
- **OnSRAM: hold off** on adding filter-budget modeling unless/until the
  primary source confirms the paper's main algorithm actually does it.
  Current best guess (secondary notes only) is that it doesn't, in which
  case OnSRAM's overflow finding above is correct behavior to leave alone.
- **The engine itself never gets a new "live" capacity-check-and-decide
  mechanism**, regardless of what the papers say — SCALE-Sim only ever
  plays back decisions an algorithm already made (see the `'P'`-triggers-
  skip pattern); inventing an eviction policy inside the engine would mean
  simulating a decision neither paper actually proposed.

## Open question — blocks starting the COSMA implementation

Extending `cosma_Ilp.py`'s tensor set `A` to include weights hits one real
mechanical gap the paper's text doesn't spell out: a normal activation's
`C[a,t]` happens one timestep *before* it's ever needed as an input
(Eq. 5 requires `P[b,t]+R[b,t]` — already resident — at the consuming
operator's own timestep, not just-created). A weight has no earlier
producer operator; it's only ever used by the one operator it belongs to.
Encoding a weight's `C[a,t]` at that operator's own timestep makes Eq. 5
fail as literally written (the weight is only `C`, not yet `P`/`R`, at
that instant).

Proposed fix: relax Eq. 5 to also accept `C[b,t]` for weight-type inputs
specifically (available the instant it's created, no producer→consumer
handoff needed, unlike activations which always have that gap by
construction). This is a disclosed interpretation, not something in the
paper's text — asked the user to either confirm this reading or point to
the paper's own supplementary material/code if it resolves the case
differently. **No answer yet as of end of session.**

## Next steps, in priority order

1. **Get an answer to the open question above** (or the OnSRAM PDF, or
   both) before writing any COSMA ILP code — this is the actual blocker.
2. **Implement COSMA activation+parameter tracking**, once (1) is
   resolved. Scoped to 5 pieces:
   - `graph_builder.load_graph()`: new `include_parameters` mode, tracking
     weight/bias tensors as real `Tensor` entries (needs a producer/
     scheduling-timestep convention — this is exactly the open question).
   - `cosma_Ilp.py`: mostly reuses existing Eq. 1-11 once `A` is bigger;
     Eq. 5's `in(a)` dependency check needs the relaxation above.
   - `baseline._make_memory_system()`: wire `filter_buf_class` the same
     way `ifmap_buf_class` already works — check
     `resident_action.get((filter_id, t)) == 'P'`, same pattern, no new
     logic needed (a weight's `'C'` timestep was never going to trigger a
     skip anyway, so its compulsory first fetch gets charged for free,
     automatically).
   - New CLI/API plumbing (e.g. `--track-parameters`) through
     `run_cosma.py`, `build_cosma_model()`, `run_baseline()`/
     `run_cosma_aware()` — same opt-in-mode pattern as `free_schedule`.
   - **Rework the budget-vs-working-set check** (both files, see above) —
     its "filter bytes are always extra, never double-counted" assumption
     is only true *because* weights aren't tracked yet. Once they are, a
     resident weight's bytes would show up in both `occupied_bytes()` and
     the filter-size term — same double-counting ambiguity already
     flagged for ifmap, now also applying to filter. Don't ship parameter
     tracking without also fixing this, or the diagnostic starts lying.
3. **Get the OnSRAM PDF if available** — resolves whether the OnSRAM
   filter-overflow finding needs the same treatment or should stay as a
   faithful limitation.
4. **Lower priority**: consider porting SMM/Zouzoula's code from `sim-opt`
   into this branch's directory structure — its paper fidelity is now
   confirmed (filters explicitly in-budget) and its model roster exists,
   but the actual port hasn't started.

## Uncommitted files as of this log (working tree, `OnSram` branch)

```
M  cosma/docs/results_plan.md          (COSMA weight-tracking note reconciled)
M  cosma/helpers/baseline.py            (budget-vs-working-set check added)
M  onsram/onsram_helpers/scale_sim_runner.py  (same check, mirrored)
?? onsram/docs/onsram_model_roster.md   (new)
?? smm/                                 (new: smm/docs/smm_model_roster.md only)
```
Nothing committed — that's deliberate, commits are yours to make.
