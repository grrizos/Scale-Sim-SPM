"""
smm_policy_selector.py
======================
Python port of the core SMM C++ logic (policy.h + manager.h) so it can be
used directly by SCALE-Sim without a C++ build step.

Policies (Section 3.2 of Zouzoula et al., ICPP '24):
  Intra  – entire layer fits on-chip
  P1     – ifmap reuse   (sliding-window ifmap, all filters resident)
  P2     – filter reuse  (whole ifmap resident, one filter at a time)
  P3     – per-channel   (one channel of every filter, whole ofmap resident)
  P4     – partial P1    (P1 with n-filter blocks; ifmap re-streamed ceil(Fn/n) times)
  P5     – partial P3    (P3 with n-filter blocks)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Policy(Enum):
    Intra = auto()
    P1 = auto()
    P2 = auto()
    P3 = auto()
    P4 = auto()
    P5 = auto()


POLICY_NAMES = {
    Policy.Intra: "intra-layer",
    Policy.P1:    "policy1 (ifmap-reuse)",
    Policy.P2:    "policy2 (filter-reuse)",
    Policy.P3:    "policy3 (per-channel)",
    Policy.P4:    "policy4 (partial-ifmap)",
    Policy.P5:    "policy5 (partial-per-ch)",
}

ALL_POLICIES = list(Policy)


def is_partial(p: Policy) -> bool:
    return p in (Policy.P4, Policy.P5)


@dataclass
class LayerSpec:
    """
    Mirrors smm::Layer (layer.h).  Fill this from SCALE-Sim topology data.

    SCALE-Sim topology columns (CSV):
        Layer name, IFMAP Height, IFMAP Width,
        Filter Height, Filter Width,
        Channels (CI), Num Filters (Fn),
        Stride, [Padding]

    OH = floor((IH + 2P - FH) / S) + 1
    OW = floor((IW + 2P - FW) / S) + 1
    CO = Fn
    """
    name: str
    IH: int; IW: int        # ifmap spatial dims
    FH: int; FW: int        # filter spatial dims
    CI: int                 # input channels
    Fn: int                 # number of filters (= output channels)
    S: int  = 1             # stride
    P: int  = 0             # padding

    @property
    def OH(self) -> int:
        return (self.IH + 2*self.P - self.FH) // self.S + 1

    @property
    def OW(self) -> int:
        return (self.IW + 2*self.P - self.FW) // self.S + 1

    @property
    def CO(self) -> int:
        return self.Fn

    def ifmap_elems(self) -> int:  return self.IH * self.IW * self.CI
    def filter_elems(self) -> int: return self.FH * self.FW * self.CI * self.Fn
    def ofmap_elems(self) -> int:  return self.OH * self.OW * self.CO
    def macs(self) -> float:
        return float(self.OH * self.OW * self.CO * self.FH * self.FW * self.CI)


@dataclass
class HwParams:
    """Mirrors smm::HwParams (manager.h)."""
    bytes_per_elem:     int   = 1     # 8-bit
    mac_per_cycle:      float = 256.0 # 16x16 PE array
    bw_bytes_per_cycle: float = 16.0  # off-chip bandwidth


@dataclass
class LayerPlan:
    policy:   Policy
    n:        int   = 0       # filter-block size (P4/P5 only)
    prefetch: bool  = False
    memory:   int   = 0       # GLB footprint in bytes
    accesses: int   = 0       # off-chip bytes
    latency:  float = 0.0     # cycles
    feasible: bool  = False

    # Partition sizes (bytes) – used to size SCALE-Sim buffers
    ifmap_bytes:  int = 0
    filter_bytes: int = 0
    ofmap_bytes:  int = 0


# ---------------------------------------------------------------------------
# Policy footprint  (memory_elems from policy.h)
# ---------------------------------------------------------------------------

def _memory_elems(p: Policy, L: LayerSpec, n: int = 0) -> int:
    IH, IW = L.IH, L.IW
    FH, FW = L.FH, L.FW
    CI, Fn = L.CI, L.Fn
    OH, OW, CO = L.OH, L.OW, L.CO

    if p == Policy.Intra:
        return L.ifmap_elems() + L.filter_elems() + L.ofmap_elems()
    if p == Policy.P1:
        return FH*IW*CI + FH*FW*CI*Fn + OW*CO
    if p == Policy.P2:
        return IH*IW*CI + FH*FW*CI + OH*OW
    if p == Policy.P3:
        return FH*IW + FH*FW*Fn + OH*OW*CO
    if p == Policy.P4:
        return FH*IW*CI + FH*FW*CI*n + OW*n
    if p == Policy.P5:
        return FH*IW + FH*FW*n + OH*OW*n
    raise ValueError(p)


def _tile_bytes(p: Policy, L: LayerSpec, n: int, bpe: int):
    """Returns (ifmap_bytes, filter_bytes, ofmap_bytes) for a policy."""
    IH, IW = L.IH, L.IW
    FH, FW = L.FH, L.FW
    CI, Fn = L.CI, L.Fn
    OH, OW, CO = L.OH, L.OW, L.CO
    b = bpe

    if p == Policy.Intra:
        return L.ifmap_elems()*b, L.filter_elems()*b, L.ofmap_elems()*b
    if p == Policy.P1:
        return FH*IW*CI*b, FH*FW*CI*Fn*b, OW*CO*b
    if p == Policy.P2:
        return IH*IW*CI*b, FH*FW*CI*b, OH*OW*b
    if p == Policy.P3:
        return FH*IW*b, FH*FW*Fn*b, OH*OW*CO*b
    if p == Policy.P4:
        return FH*IW*CI*b, FH*FW*CI*n*b, OW*n*b
    if p == Policy.P5:
        return FH*IW*b, FH*FW*n*b, OH*OW*n*b
    raise ValueError(p)


def _choose_block(p: Policy, L: LayerSpec, glb_elems: int) -> int:
    """Largest n in [1, Fn] whose working set fits glb_elems. 0 = infeasible."""
    best = 0
    for n in range(1, L.Fn + 1):
        if _memory_elems(p, L, n) <= glb_elems:
            best = n
        elif n > 1:
            break   # monotonically increasing
    return best


def _min_memory_elems(p: Policy, L: LayerSpec) -> int:
    return _memory_elems(p, L, 1) if is_partial(p) else _memory_elems(p, L, 0)


# ---------------------------------------------------------------------------
# Off-chip access estimate  (estimate_accesses from policy.h)
# ---------------------------------------------------------------------------

def _estimate_accesses(p: Policy, L: LayerSpec, n: int, bpe: int) -> int:
    reload = math.ceil(L.Fn / n) if is_partial(p) else 1
    elems  = L.ifmap_elems() * reload + L.filter_elems() + L.ofmap_elems()
    return elems * bpe


# ---------------------------------------------------------------------------
# Evaluate one (policy, prefetch) candidate  (evaluate() from manager.h)
# ---------------------------------------------------------------------------

def _evaluate(p: Policy, prefetch: bool, L: LayerSpec,
               glb_bytes: int, hw: HwParams) -> LayerPlan:
    plan = LayerPlan(policy=p, prefetch=prefetch)

    mult = 2 if prefetch else 1
    glb_elems_eff = (glb_bytes // hw.bytes_per_elem) // mult

    n = 0
    if is_partial(p):
        n = _choose_block(p, L, glb_elems_eff)
        if n == 0:
            return plan  # infeasible
    elif _min_memory_elems(p, L) > glb_elems_eff:
        return plan      # infeasible

    plan.n        = n
    ib, fb, ob   = _tile_bytes(p, L, n, hw.bytes_per_elem)
    plan.ifmap_bytes  = ib * mult
    plan.filter_bytes = fb * mult
    plan.ofmap_bytes  = ob * mult
    plan.memory       = plan.ifmap_bytes + plan.filter_bytes + plan.ofmap_bytes

    plan.accesses = _estimate_accesses(p, L, n, hw.bytes_per_elem)

    compute_cyc  = L.macs() / hw.mac_per_cycle
    transfer_cyc = plan.accesses / hw.bw_bytes_per_cycle
    plan.latency = (max(compute_cyc, transfer_cyc)
                    if prefetch else compute_cyc + transfer_cyc)
    plan.feasible = True
    return plan


# ---------------------------------------------------------------------------
# Algorithm 1: best policy for a single layer  (best_for_layer from manager.h)
# ---------------------------------------------------------------------------

class Objective(Enum):
    Accesses = auto()
    Latency  = auto()


def best_for_layer(L: LayerSpec, glb_bytes: int, hw: HwParams,
                   objective: Objective = Objective.Accesses,
                   allow_prefetch: bool = True) -> LayerPlan:
    """
    Returns the optimal LayerPlan for L given a GLB of glb_bytes.
    If no policy fits, returns a LayerPlan with feasible=False.
    """
    best: Optional[LayerPlan] = None
    best_metric = float('inf')

    for p in ALL_POLICIES:
        for pf in ([False, True] if allow_prefetch else [False]):
            cand = _evaluate(p, pf, L, glb_bytes, hw)
            if not cand.feasible:
                continue
            m = cand.accesses if objective == Objective.Accesses else cand.latency
            if m < best_metric:
                best_metric = m
                best = cand

    return best if best is not None else LayerPlan(policy=Policy.P2)


# ---------------------------------------------------------------------------
# Heterogeneous / homogeneous scheme  (Section 3.3)
# ---------------------------------------------------------------------------

def plan_network(layers: List[LayerSpec],
                 glb_bytes: int,
                 hw: HwParams,
                 objective: Objective = Objective.Accesses,
                 allow_prefetch: bool  = True,
                 homogeneous: bool     = False) -> List[LayerPlan]:
    """
    Returns a per-layer list of LayerPlan (heterogeneous by default).
    Set homogeneous=True to force one policy for all layers.
    """
    if not homogeneous:
        return [best_for_layer(L, glb_bytes, hw, objective, allow_prefetch)
                for L in layers]

    # Homogeneous: one policy that works for every layer, minimum total objective
    best_plans: List[LayerPlan] = []
    best_total = float('inf')

    for p in ALL_POLICIES:
        for pf in ([False, True] if allow_prefetch else [False]):
            plans = []
            ok = True
            total = 0.0
            for L in layers:
                plan = _evaluate(p, pf, L, glb_bytes, hw)
                if not plan.feasible:
                    ok = False; break
                plans.append(plan)
                total += plan.accesses if objective == Objective.Accesses else plan.latency
            if ok and total < best_total:
                best_total = total
                best_plans = plans

    return best_plans


# ---------------------------------------------------------------------------
# Helper: build a LayerSpec from a SCALE-Sim topology row dict
# ---------------------------------------------------------------------------

def layer_spec_from_scalesim_row(name: str, row: dict) -> LayerSpec:
    """
    row keys (case-insensitive) as produced by scalesim topology_utils:
        IFMAP Height, IFMAP Width, Filter Height, Filter Width,
        Channels, Num Filter, Strides, [Padding]
    """
    def g(k): return int(row.get(k, row.get(k.lower(), 0)))
    return LayerSpec(
        name = name,
        IH   = g('IFMAP Height'),
        IW   = g('IFMAP Width'),
        FH   = g('Filter Height'),
        FW   = g('Filter Width'),
        CI   = g('Channels'),
        Fn   = g('Num Filter'),
        S    = g('Strides') or 1,
        P    = g('Padding'),
    )
