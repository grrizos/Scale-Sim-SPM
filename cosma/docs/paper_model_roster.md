# Paper Model Roster: what's available to test against arXiv:2311.18246

Tracks which of the COSMA paper's own 14 evaluation models (§V-A1) we can
actually run, so results are checked against real availability, not
assumed. Checked directly against `trim/models/` and `cosma/_exported/` on
2026-09-10 — update this file as models get sourced/exported.

## 1. Paper's own roster (§V-A1)

10 human-designed DNNs (Figure 3) across 4 application domains, plus 4
NAS-generated DNNs (Figure 4, Table II) needing COSMA's divide-and-conquer
heuristic to solve at all.

| Model | Paper category | Available now? | Notes |
|---|---|---|---|
| **ResNet-50** | human-designed (image classification) | ✅ `_exported/_exported_resnet50-tflite-float/model.json` (auto-exported from `_exported/resnet50-tflite-float/resnet50.tflite`) | Ready to run today |
| **DenseNet** | human-designed (image classification) | ✅ `_exported/_exported_densenet121-tflite-float/model.json` | Assumed DenseNet-121 — paper doesn't specify the variant |
| **DeepLabV3** | human-designed (semantic segmentation) | ✅ `_exported/deeplabv3/DeepLabV3-Plus-MobileNet.tflite` (not yet auto-exported to model.json) | Paper itself flags this one as slower to solve (6.96s/17.242s at `M_R`/`M_H`) |
| ResNeXt | human-designed (image classification) | ❌ not sourced | Would need exporting first |
| R2Plus1D | human-designed (video classification) | ❌ not sourced | Video model — would need exporting first |
| S3D | human-designed (video classification) | ❌ not sourced | Video model — would need exporting first |
| FCN | human-designed (semantic segmentation) | ❌ not sourced | The paper's own one documented exception where all 4 baselines tie with COSMA at `M_P` — worth sourcing specifically to check this |
| L-RASPP | human-designed (semantic segmentation) | ❌ not sourced | Would need exporting first |
| Transformer | human-designed (transformer-based) | ❌ not sourced | Would need exporting first |
| ViT | human-designed (transformer-based) | ❌ not sourced | Would need exporting first |
| PNASNet-5 | NAS-generated | ⛔ permanently out of scope | Needs the divide-and-conquer heuristic — standing project decision to never build this |
| AmoebaNet-D | NAS-generated | ⛔ permanently out of scope | Same |
| NASNet-A | NAS-generated | ⛔ permanently out of scope | Same |
| DARTS | NAS-generated | ⛔ permanently out of scope | Same |

## 2. Summary

**3 of 14 are ready to run right now**: ResNet-50, DenseNet-121, DeepLabV3 —
directly matching the paper's own models. The other 7 human-designed models
would need sourcing/exporting before they're usable. The 4 NAS-generated
models are off the table regardless of availability, per the standing
decision to skip the divide-and-conquer heuristic (see `results_plan.md`).

## 3. How to check numbers against the paper for an available model

1. Get `M_R`/`M_H`/`M_P` first (instant, no simulation):
   ```
   PYTHONPATH=..:. python3 visualize_spm.py --model-json <path> --bounds-only --true-mpmf --solver gurobi
   ```
2. Run all 4 paper baselines + COSMA at those budgets:
   ```
   PYTHONPATH=..:. python3 run_paper_baselines.py --model-json <path> \
       --budgets-kb <M_R> <M_H> <M_P> --solver gurobi \
       --out-csv results/<name>_paper_baselines.csv
   ```
   See `run_paper_baselines.py` / `docs/baseline_construction.md`.

## 4. Caveats on "matching numbers"

- The paper reports its actual reduction numbers as bar charts (Fig 3/4),
  not a numeric table — the only exact figures given in text are the
  aggregate averages (84% for human-designed models at `M_R`, 85% for NAS
  via divide-and-conquer), not clean per-model targets. "Matching" means
  matching the qualitative pattern (COSMA ≤ baselines always, gap biggest
  at `M_R` and shrinking toward `M_P`, greedy ≤ Belady), not reproducing
  exact byte counts.
- Different array config (this project's `configs/scale.cfg` vs. the
  paper's own hardware assumptions), a different exact PyTorch→TFLite
  export pipeline, and a reimplementation of the paper's baselines from its
  textual description (not their original code) all mean exact numeric
  match was never the realistic goal — see `docs/results_plan.md` §6 for
  the other acknowledged, permanent differences (solver/hardware).
- **First real test** (ResNet-50 @ ~9420KB, close to `M_P`, 2026-09-10): all
  5 schemes tied at 0 non-compulsory bytes. Consistent with the paper's own
  guarantee that COSMA reaches 0 at `M_P`, but the paper's text implies the
  4 baselines should still show nonzero traffic at `M_P` for every model
  *except* FCN — and ResNet-50 isn't documented as an FCN-like exception.
  Not necessarily wrong (budget was right at the edge — only ~12KB of slack
  above peak occupancy — and the default schedule's own peak may just equal
  `M_P` here, same pattern already seen on the small DenseNet fixture), but
  flagged as open, not confirmed. Worth re-checking at `M_R`/`M_H` instead,
  where the paper's own charts show the real, non-degenerate gap — not yet
  done.
