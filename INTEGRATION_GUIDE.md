# SMM × SCALE-Sim Integration Guide

Integrates the scratchpad memory management policies from
**Zouzoula et al., ICPP '24** into SCALE-Sim so every simulated layer
uses the analytically-optimal GLB partition.

---

## File overview

| File | Role |
|---|---|
| `smm_policy_selector.py` | Pure-Python port of `policy.h` + `manager.h` |
| `smm_scalesim_runner.py` | Drop-in SCALE-Sim runner with SMM pre-planning |
| `test_smm_integration.py` | Validates against paper Table 3; shows the sweep |

---

## How the integration works

```
┌─────────────────────────────────────────────────────┐
│                 SMMScaleSimRunner.run()              │
│                                                     │
│  1. Build LayerSpec list from SCALE-Sim topology    │
│                                                     │
│  2. SMM Algorithm 1 (plan_network)                  │
│     For every layer pick the best policy P1-P5:     │
│       • memory_elems(policy, layer, n) ≤ GLB/2      │  ← feasibility
│       • minimise off-chip accesses (or latency)     │  ← objective
│     Result: per-layer LayerPlan with                │
│       ifmap_bytes / filter_bytes / ofmap_bytes      │
│                                                     │
│  3. For each layer                                  │
│     a) Construct double_buffered_scratchpad         │
│        with SMM-dictated buffer sizes               │  ← KEY INJECTION
│     b) single_layer_sim.set_memory_system(mem)      │
│     c) single_layer_sim.run()   ← normal SCALE-Sim │
└─────────────────────────────────────────────────────┘
```

### The key injection (step 3a)

SCALE-Sim's `single_layer_sim` has `set_memory_system()` — it accepts a
pre-configured `double_buffered_scratchpad` and skips its own buffer sizing.
SMM exploits exactly this hook:

```python
mem.set_params(
    ifmap_buf_size_bytes  = plan.ifmap_bytes,   # ← SMM-computed
    filter_buf_size_bytes = plan.filter_bytes,  # ← SMM-computed
    ofmap_buf_size_bytes  = plan.ofmap_bytes,   # ← SMM-computed
    ...
)
sim.set_memory_system(mem)
```

---

## Policies and what they control

| Policy | ifmap tile | filter tile | ofmap tile | reloads |
|--------|-----------|-------------|-----------|---------|
| Intra  | full ifmap | full filters | full ofmap | ×1 |
| P1 ifmap-reuse | FH·IW·CI (sliding) | all filters | OW·CO | ×1 |
| P2 filter-reuse | full ifmap | one filter | OH·OW | ×1 |
| P3 per-channel | FH·IW (1ch) | FH·FW·Fn (1ch each) | full ofmap | ×1 |
| P4 partial P1 | sliding window | FH·FW·CI·n | OW·n | ×⌈Fn/n⌉ |
| P5 partial P3 | FH·IW (1ch) | FH·FW·n | OH·OW·n | ×⌈Fn/n⌉ |

Algorithm 1 evaluates all 6 + their double-buffered variants (×2 buffer
space) and picks the one whose working set fits in the GLB and minimises
the objective.

---

## Quick start``

### Run validation (no topology file needed)

```bash
python test_smm_integration.py
```

Expected output (matches paper Table 3):
```
intra-layer    2353.0 kB   ✓
P1 (filters)   2318.0 kB   ✓
P2 (ifmap)      199.6 kB   ✓
P3 (ofmap)      788.6 kB   ✓
```

### Run with your own topology

```bash
python3 smm_scalesim_runner.py \
  --topology topologies/resnet18.csv \
  --config   configs/scale.cfg \
  --glb_kb   64 \
  --objective accesses \
  --output   outputs/smm_64k
```

### 5. Use as a library

```python
from smm_scalesim_runner import SMMScaleSimRunner

runner = SMMScaleSimRunner(
    topology_file  = "topologies/resnet18.csv",
    config_file    = "configs/scale.cfg",
    glb_size_kb    = 64,
    objective      = "accesses",   # or "latency"
    homogeneous    = False,        # True = one policy for all layers
    allow_prefetch = True,
    output_dir     = "outputs/smm_run",
)
runner.run()
runner.print_summary()
```

---

## Configuration knobs

| Parameter | Effect |
|-----------|--------|
| `glb_size_kb` | Total on-chip GLB in kB. SMM partitions this per policy. |
| `objective` | `"accesses"` minimises DRAM traffic; `"latency"` minimises cycles. |
| `homogeneous` | `False` (default) = different policy per layer; `True` = one policy for all. |
| `allow_prefetch` | `True` (default) = considers double-buffered variants (halves available GLB but hides DRAM latency). |

---

## Expected benefit (ResNet18, 8-bit)

| GLB | Homogeneous | Heterogeneous | Saving |
|-----|------------|--------------|--------|
| 64 kB  | 20.28 MB | 16.07 MB | **20.8 %** |
| 128 kB | 17.22 MB | 15.59 MB | 9.4 % |
| 256 kB | 15.59 MB | 15.59 MB | 0 % |

---

## Extending to other networks

Add your network as a list of `LayerSpec` objects and pass to `plan_network()`:

```python
from smm_policy_selector import LayerSpec, HwParams, Objective, plan_network

layers = [
    LayerSpec("layer0", IH=56, IW=56, FH=3, FW=3, CI=64, Fn=64, S=1, P=1),
    LayerSpec("layer1", IH=28, IW=28, FH=3, FW=3, CI=64, Fn=128, S=2, P=1),
    # ...
]
hw    = HwParams(bytes_per_elem=1, mac_per_cycle=256, bw_bytes_per_cycle=16)
plans = plan_network(layers, glb_bytes=64*1024, hw=hw)
for spec, plan in zip(layers, plans):
    print(spec.name, plan.policy, plan.accesses // 1024, "kB")
```

---

## Notes

- **No C++ build required** — `smm_policy_selector.py` is a pure-Python port.
- **No LRU / cache model** — the SMM paper uses a software-managed scratchpad
  with closed-form off-chip estimates; no eviction logic is needed or modelled.
- **Depth-wise convolution** — P3/P5 need group-aware filter sizing; not
  handled here (same scope as the original C++ repo).

## Data flow summary
```bash
topology CSV ──► _build_layer_specs() ──► [LayerSpec, ...]
                                               │
config.cfg ──► HwParams                        │
                    └──────────────────► plan_network()  (SMM Algorithm 1)
                                               │
                                         [LayerPlan, ...]
                                          ifmap/filter/ofmap bytes
                                               │
                                     _make_memory_system()
                                               │
                                     sim.set_memory_system()
                                               │
                                          sim.run()  (SCALE-Sim cycle sim)
                                               │
                                        traces/ + print_summary()
```