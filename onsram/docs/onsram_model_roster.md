# Paper Model Roster: what's available to test against the OnSRAM paper

Tracks which of the OnSRAM paper's own 12 evaluation networks (Abstract, Table 1) we can
actually run, so results are checked against real availability, not assumed. Same format/purpose
as `cosma/docs/paper_model_roster.md` — checked directly against `cosma/_exported/` (the shared
export cache both papers read from, see `spm_common/model_resolver.py`) and `trim/models/` on
2026-09-16.

## 1. Paper's own roster

12 real networks, evaluated at 3 TFLOP / 2MB SPM / 32 GBps / batch size 1 (per
`onsram_integration_plan.md` §4's paper-fidelity check).

| Model | Available now? | Notes |
|---|---|---|
| **VGG-16** | ✅ `_exported/VGG16/model.json` | Ready to run today |
| **Inception-v3** | ✅ `_exported/_exported_inception_v3-tflite-float/model.json` | Ready to run today (the raw `.tflite` also sits at `_exported/inception_v3-tflite-float/`, under a different export name) |
| **ResNet-50** | ✅ `_exported/_exported_resnet50-tflite-float/model.json` | Ready to run today — same exported artifact COSMA's own roster already uses |
| **MobileNetV1** | ✅ `_exported/MobileNet/model.json` | Ready to run today — this is `run_onsram.py`'s own default model (`DEFAULT_MODEL = 'MobileNet'`) |
| **SqueezeNet** | ✅ `_exported/squeezenet/model.json` | Ready to run — but `_exported/squeezenet_small_cifar100_int8_tucker_svd_5/` is a separate, quantized/reduced variant; paper doesn't specify which SqueezeNet |
| AlexNet | ❌ not sourced | Would need exporting first |
| GoogLeNet | ❌ not sourced | Would need exporting first |
| Inception-v4 | ❌ not sourced | Only v3 is currently exported |
| SSD300 | ❌ not sourced | Detection model — would need exporting first |
| ResNeXt | ❌ not sourced | Same gap already flagged in COSMA's own roster |
| **PTB-LSTM** | ⛔ structurally blocked | Recurrent, not conv-like — not just unsourced: even exported, it wouldn't get real simulated cost today (see §3) |
| **Multi-Head Attention** | ⛔ structurally blocked | Same reason — attention's matmuls *could* be represented via a GEMM-as-1×1-conv extension, but that work doesn't exist yet |

## 2. Summary

**5 of 12 ready to run right now**: VGG-16, Inception-v3, ResNet-50, MobileNetV1, SqueezeNet —
directly matching the paper's own models, and a noticeably better hit rate than COSMA's 3/14,
mostly because OnSRAM's model list overlaps heavily with artifacts already exported for COSMA
(same shared `_exported/` cache, see `spm_common/__init__.py`). 5 more (AlexNet, GoogLeNet,
Inception-v4, SSD300, ResNeXt) are a plain sourcing gap — export and they're runnable, no
different from COSMA's own "not sourced" entries. The remaining 2 (PTB-LSTM, Multi-Head
Attention) are a *different kind* of gap: not about sourcing a `.tflite`, but about
`topology_builder.py`/`onsram_helpers/topology.py` only generating a real topology row for
`CONV2D`/`DEPTHWISE_CONV2D` — recurrent and attention layers would need real engine-adjacent work
(the GEMM-as-1×1-conv topology-row trick discussed for COSMA's own Transformer/ViT gap) before
they'd produce trustworthy numbers, not just an export step.

## 3. How to check numbers against the paper for an available model

```bash
cd onsram
python3 run_onsram.py --model VGG16 --spm-mb 2
python3 run_onsram.py --model _exported_inception_v3-tflite-float --spm-mb 2
python3 run_onsram.py --model _exported_resnet50-tflite-float --spm-mb 2
python3 run_onsram.py --model MobileNet --spm-mb 2        # the built-in default
python3 run_onsram.py --model squeezenet --spm-mb 2
```
`--model` accepts a bare name under `cosma/_exported/<name>/model.json` (see `resolve_model_arg()`
in `run_onsram.py`) — same convention as the folder names in the table above.

## 4. Caveats on "matching numbers"

- The paper reports headline numbers as **1.02–4.8× latency reduction** (Static), averaged across
  all 12 models at 2MB/32GBps — not a clean per-model target table, so "matching" means the same
  qualitative pattern (larger reduction on models with more reusable activation traffic, smaller
  on already-compute-bound ones), not reproducing an exact per-model number.
- Different array config (this project's `configs/scale.cfg` vs. the paper's own 3 TFLOP/32GBps
  hardware assumption), a different `.tflite` export pipeline than whatever the paper's own
  authors used, and this port's own physically-corrected inclusive-lifetime pinning (vs. the
  reference implementation's exclusive-end bug — see `onsram_helpers/pinning.py`'s docstring) all
  mean exact numeric match was never the goal here, same acknowledgment COSMA's own roster makes.
- Not yet run against any of the 5 available models except MobileNet@2MB (see this project's own
  logs under `onsram/logs/`) — VGG-16/Inception-v3/ResNet-50/SqueezeNet numbers are unverified,
  just confirmed *runnable*.
