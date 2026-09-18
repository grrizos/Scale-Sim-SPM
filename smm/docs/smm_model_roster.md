# Paper Model Roster: what's available to test against the SMM paper

Tracks which of the SMM paper's own 6 evaluation models (Table 2) we can actually run, so results
are checked against real availability, not assumed. Same format/purpose as
`cosma/docs/paper_model_roster.md` and `onsram/docs/onsram_model_roster.md` — checked directly
against `cosma/_exported/` on 2026-09-16.

**Paper**: Stavroula Zouzoula, Mohammad Ali Maleki, Muhammad Waqar Azhar, Pedro Trancoso.
"Scratchpad Memory Management for Deep Learning Accelerators." ICPP '24.
https://doi.org/10.1145/3673038.3673115

**Not yet ported to this branch** — `smm_policy_selector.py` (the ported policy-selection logic,
Intra/P1-P5) and `smm_scalesim_runner.py` (the SCALE-Sim driver) currently live on the `sim-opt`
branch, as loose top-level files, not under a `smm/` directory the way COSMA/OnSRAM are organized
here. This roster only tracks model *availability* ahead of that port — it doesn't imply the
implementation exists on this branch yet.

## 1. Paper's own roster (Table 2)

6 models, all CNN/image-classification, described by layer-type composition: CV (conv), DW
(depthwise conv), PW (pointwise/1×1 conv), FC (fully-connected), PL (projection layer — the 1×1
shortcut conv in a residual block).

| Model | Layers (paper) | Layer types | Available now? | Notes |
|---|---|---|---|---|
| **MobileNet** | 28 | CV, DW, PW, FC | ✅ `_exported/MobileNet/model.json` | Ready to run today — same artifact OnSRAM's roster uses as "MobileNetV1" |
| **MobileNetV2** | 53 | CV, DW, PW, FC | ✅ `_exported/MobileNetV2/model.json` | Ready to run today |
| **EfficientNetB0** | 82 | CV, DW, PW, FC | ⚠️ `_exported/efficient50/model.json` | An EfficientNet variant is exported (confirmed via its ops: SIGMOID+MUL Swish-activation pattern, 243 layers) — but the folder name ("efficient50") doesn't confirm it's specifically B0 vs. a different compound-scaling variant. Same "paper doesn't pin the exact variant" caveat COSMA's roster already makes for DenseNet. |
| GoogLeNet | 64 | CV, PW, FC | ❌ not sourced | Same gap OnSRAM's own roster already flags for this model |
| MnasNet | 53 | CV, DW, PW, FC | ❌ not sourced | Would need exporting first |
| ResNet18 | 21 | CV, PW, FC, PL | ❌ not sourced | We have ResNet-50 and ResNet-20-CIFAR10 exported, not ResNet-18 specifically |

## 2. Summary

**2 of 6 ready to run right now** (MobileNet, MobileNetV2), **1 more likely available but
variant-unconfirmed** (EfficientNetB0-ish), **3 not sourced** (GoogLeNet, MnasNet, ResNet-18) —
GoogLeNet's gap is now shared across all three paper rosters in this repo (COSMA doesn't list it
at all, OnSRAM and SMM both want it, neither has it).

**Every one of this paper's 6 models includes FC (fully-connected) layers** — unlike COSMA/OnSRAM,
where FC-head coverage varies per model, the SMM paper's Table 2 lists FC for all 6. That makes
DENSE-as-1×1-conv support (discussed for COSMA/OnSRAM) the single highest-leverage engine addition
for this paper specifically — without it, every one of these 6 models' classifier head is
zero-cost, understating both papers' latency and DRAM totals for the same reason (see the
model-coverage discussion in this conversation).

## 3. Config note (for matching the paper's own baseline, once ported)

Paper's own baseline setup (§4): 16×16 PE array, **output-stationary** dataflow, 8-bit data width,
16 elements/cycle off-chip bandwidth, batch size 1, ofmap buffer fixed at 4KB with the remaining
budget split ifmap/filter at 25-75%/50-50%/75-25% (three separate baseline configs, not one) —
GLB sizes tested: 64KB, 128KB, 256KB, 512KB, 1MB. None of this matches `configs/scale.cfg`'s
current defaults (which are WS dataflow, per the earlier architecture discussion) — worth a
dedicated config file (mirroring `configs/scale_onsram.cfg`'s own precedent) once this paper is
actually ported here, not reusing COSMA/OnSRAM's shared `configs/scale.cfg`.
