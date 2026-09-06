# COSMA + SCALE-Sim Integration — Project Plan

---

## 1. What Are We Trying To Do

We are implementing the **COSMA algorithm** (Combined Scheduling, Memory Allocation and Tensor Replacement) from the Princeton paper (arXiv 2023) and connecting it to **SCALE-Sim**, a cycle-accurate systolic array simulator.

The goal is to use COSMA to **optimize how a DNN executes on a hardware accelerator with limited on-chip scratchpad memory (SPM)**, then use SCALE-Sim to **measure the actual cycle-level impact** of that optimization.

### The Problem COSMA Solves

When a DNN runs on an accelerator, its tensors (activations) need to live somewhere between layers. If the SPM is too small to hold all of them simultaneously, some tensors must be written to DRAM and read back later. This extra data movement is called **non-compulsory DRAM traffic** and it is expensive — DRAM is 50–100× slower than on-chip memory.

COSMA minimizes this non-compulsory traffic by jointly optimizing three things:

1. **Operator Schedule** — what order do the layers execute in?
2. **Memory Allocation** — where exactly in the SPM does each tensor live?
3. **Tensor Replacement** — which tensors get spilled to DRAM and when?

It does this using **Integer Linear Programming (ILP)** — a mathematical solver that finds the globally optimal combination of all three decisions simultaneously.

### What SCALE-Sim Contributes

SCALE-Sim runs cycle-accurate simulation of a systolic array accelerator. It gives us:
- **Compute cycles** per layer (how long the MAC array takes)
- **Compulsory DRAM bytes** per layer (minimum data movement just to run the operator)
- **SRAM access traces** per layer

COSMA then adds the **non-compulsory DRAM traffic** (spills and retrievals) on top of SCALE-Sim's compulsory costs, giving the full picture.

### What We Are NOT Doing

- We are NOT modifying SCALE-Sim's internal simulation engine
- We are NOT optimizing intra-layer tiling (COSMA treats each layer as a black box)
- We are NOT implementing OnSRAM or SPM-for-DL (those are separate papers)

---

## 2. Why This is Hard

### SCALE-Sim Does Not Understand Inter-Layer Memory

SCALE-Sim simulates each layer **independently**. When a layer finishes, SCALE-Sim forgets everything. It has no concept of:
- A tensor surviving from layer N to layer N+3
- SPM being shared across multiple layers
- Spilling a tensor to DRAM to make room for another

SCALE-Sim has **three fixed separate buffers** per layer (ifmap, filter, ofmap). COSMA needs a **single unified SPM address space** where all tensors from all layers coexist simultaneously.

### The ILP Grows Quadratically

COSMA's memory allocation variables (u[a,b,t] and d[a,b,t]) are created for every **pair** of tensors that could be in SPM at the same time. For a 64-layer model this can be thousands of variables. Without careful filtering (only creating pairs whose lifetimes actually overlap), the ILP becomes unsolvable.

---

## 3. The Architecture of the Solution

```
model.json                        SCALE-Sim topology.csv
      |                                      |
      v                                      v
[graph_builder.py]              [Run SCALE-Sim baseline]
Build tensor graph:              Get per-layer costs:
- nodes (operators)              - compute_cycles
- tensors (activations)          - compulsory_dram_bytes
- producer/consumer links
      |                                      |
      +──────────────┬───────────────────────+
                     |
                     v
              [unified_spm.py]
         Unified SPM address space
         (replaces 3 fixed buffers)
                     |
                     v
              [cosma_ilp.py]
         Build ILP model:
         - Variables: C, P, S, R, L, V, u, d
         - Constraints: Eq. 1-11
         - Objective: Eq. 12 (minimize spill+retrieve)
                     |
                     v
              [Solver: Gurobi / CBC]
         Finds optimal values for all variables
                     |
                     v
         Extract results:
         - ordered_layers  (optimal execution order)
         - extra_dram[t]   (spill+retrieve bytes per timestep)
         - spm_plan[a,t]   (base address of each tensor)
                     |
                     v
              [run_cosma.py]
         Re-run SCALE-Sim in COSMA's order:
         total_cycles += max(compute_cycles,
                            (compulsory_dram + extra_dram) / BW)
                     |
                     v
              Final Output:
         - Optimized cycle count
         - DRAM traffic reduction %
         - Speedup vs baseline
```

---

## 4. The COSMA ILP — Variables and Equations

### Variables

For every tensor `a` and every timestep `t` (one timestep = one operator execution):

| Variable | Type | Meaning |
|---|---|---|
| `C[a,t]` | Binary | Tensor `a` is **created** at timestep `t` |
| `P[a,t]` | Binary | Tensor `a` is **preserved** in SPM at timestep `t` |
| `S[a,t]` | Binary | Tensor `a` is **spilled** to DRAM at timestep `t` |
| `R[a,t]` | Binary | Tensor `a` is **retrieved** from DRAM at timestep `t` |
| `L[a,t]` | Integer | **Base address** of tensor `a` in SPM at timestep `t` |
| `V[a,t]` | Binary | Tensor `a` keeps the **same address** from `t-1` to `t` |
| `u[a,b,t]` | Binary | Tensor `a` is placed **above** tensor `b` in SPM at `t` |
| `d[a,b,t]` | Binary | Tensor `a` is placed **below** tensor `b` in SPM at `t` |

### Equations

| Equation | Formula | What it enforces |
|---|---|---|
| Eq.1 | `C+P+S+R <= 1` | Only one action per tensor per timestep |
| Eq.2 | `P[a,t] <= C[a,t-1]+P[a,t-1]+R[a,t-1]` | Can only preserve if was resident before |
| Eq.3 | `S[a,t] <= C[a,t-1]+P[a,t-1]` | Can only spill if was resident before |
| Eq.4 | `R[a,t] <= sum(S[a,k] for k<=t)` | Can only retrieve if previously spilled |
| Eq.5 | `C[a,t] <= P[b,t]+R[b,t]` for all inputs b | Inputs must be in SPM when operator runs |
| Eq.6 | `C[a,t] == C[b,t]` for sibling tensors | Sibling tensors created at same timestep |
| Eq.7 | `sum(C[a,t]) == 1` | Each tensor created exactly once |
| Eq.8 | `sum(S[a,t]) <= 1` | Each tensor spilled at most once |
| Eq.9 | `L[a,t] + Size(a) <= Budget` | Tensor fits within memory budget |
| Eq.10 | non-overlap constraints via u,d | No two tensors overlap in SPM |
| Eq.11 | `L[a,t] == L[a,t-1]` when `V=1` | Pinned tensor keeps same address |
| Eq.12 | `minimize sum((S+R)*Size)` | **Objective: minimize spill+retrieve traffic** |

---

## 5. The Model We Are Working With

The current `model.json` is **MobileNetV2** trained on CIFAR-10:

- **64 layers** (CONV2D, DEPTHWISE_CONV2D, ADD)
- **64 activation tensors** tracked by COSMA
- **10 skip-connection tensors** (consumed by more than one layer — the residual adds)
- Layer 0 has no activation input (first layer, reads raw input from DRAM)

### Key tensor sizes observed

| Tensor | Size | Producer | Consumers |
|---|---|---|---|
| Tensor 3 | 16 KB | Layer 0 | Layer 1 |
| Tensor 6 | 16 KB | Layer 1 | Layer 2 |
| Tensor 9 | 8 KB | Layer 2 | Layer 3 |
| Tensor 12 | 48 KB | Layer 3 | Layer 4 |
| Tensor 18 | ~8 KB | Layer 5 | Layers 6 AND 9 (skip) |

The skip-connection tensors (like tensor 18) are the most important for COSMA — they must survive in SPM across multiple layers, which is exactly the kind of decision the ILP is designed to optimize.

---

## 6. Current Codebase

### What exists right now

```
project/
├── graph_builder.py          ✅ DONE
│   - compute_size_bytes()
│   - load_graph()            reads model.json → nodes + tensors dicts
│   - print_graph_summary()   debug output
│
├── model.json                ✅ PROVIDED
│   - 172 tensors (shapes + dtypes)
│   - 64 layers (op, inputs, inputs_from, outputs, params, weights, bias)
│
└── (coming next)
    ├── unified_spm.py        ⬜ NOT STARTED
    ├── cosma_ilp.py          ⬜ NOT STARTED
    └── run_cosma.py          ⬜ NOT STARTED
```

### What `graph_builder.py` produces

```python
nodes = {
    0: Node(id=0, op='CONV2D',
            activation_inputs=[],      # first layer, no activation input
            weight_inputs=[0, 1, 2],   # image + kernel + bias
            outputs=[3]),

    1: Node(id=1, op='DEPTHWISE_CONV2D',
            activation_inputs=[3],     # tensor 3 flows from layer 0
            weight_inputs=[4, 5],      # depthwise kernel + bias
            outputs=[6]),
    ...
}

tensors = {
    3: Tensor(id=3, size_bytes=16384,
              producer_layer=0,
              consumer_layers=[1],
              producer_timestep=-1,     # filled in during scheduling
              last_used_timestep=-1),   # filled in during scheduling

    18: Tensor(id=18, size_bytes=8192,
               producer_layer=5,
               consumer_layers=[6, 9], # skip connection!
               ...),
    ...
}
```

### Key design decisions made so far

1. **Activation tensors only** — weight/bias tensors (where `inputs_from == -1`) are excluded from COSMA's ILP. They are always compulsory DRAM accesses.
2. **Tensor id as key** — tensors are keyed by their integer id from model.json, not by name string.
3. **`inputs_from` field** is the source of truth for inter-layer dependencies — it directly maps each input tensor to the layer that produced it.

---

## 7. Next Steps (In Order)

### Step 2 — `unified_spm.py`
Build a unified SPM class that holds all tensors in one address space. Replaces SCALE-Sim's 3 fixed buffers. Tracks what is currently resident, at what address, and how much free space remains.

### Step 3 — `cosma_ilp.py` Part A: Variables
Create all C, P, S, R, L, V, u, d variables in PuLP. Apply overlap filtering to limit u/d pairs to only tensors whose liveness windows intersect.

### Step 4 — `cosma_ilp.py` Part B: Constraints
Add Eq. 1–11 to the ILP. The most complex is Eq.10 (non-overlap) due to the big-M encoding and the quadratic number of tensor pairs.

### Step 5 — `cosma_ilp.py` Part C: Objective + Solve
Add Eq.12 (minimize spill+retrieve bytes) and call Gurobi or CBC solver.

### Step 6 — Extract results
Read solver variable values → produce `ordered_layers`, `extra_dram_bytes`, `spm_plan`.

### Step 7 — `run_cosma.py`
Connect everything: run SCALE-Sim baseline, run COSMA ILP, re-simulate with COSMA's plan, compare results.

---

## 8. Key Risks and How We Handle Them

| Risk | Impact | Mitigation |
|---|---|---|
| ILP too slow for 64 layers | Solver times out | Overlap filtering on u/d pairs; use Gurobi not CBC |
| SCALE-Sim layer order is fixed | Cannot reorder | We call SCALE-Sim per-layer independently and sum results |
| Skip connections tricky | Tensor must survive multiple layers | Correctly encoded via consumer_layers list in graph |
| Big-M in Eq.10 too loose | Solver slow to converge | Set M = memory_budget, not a huge arbitrary number |
| Layer 0 has no activation input | Eq.5 loop breaks | Handled: activation_inputs=[] so loop simply doesn't execute |