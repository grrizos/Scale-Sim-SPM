# onsram/onsram_helpers/scale_sim_runner.py
"""
Drives SCALE-Sim in-process (no subprocess, no intermediate CSV round-trip)
to get per-layer compute cycles and compulsory DRAM bytes for every layer
in model.json -- a self-contained duplicate of cosma/helpers/baseline.py's
run_baseline()/run_cosma_aware(), NOT an import of them. COSMA and OnSRAM
are two separate jobs that happen to sit on the same SCALE-Sim engine; a
future change to COSMA's own baseline.py (or its own topology_builder.py /
cosma_resident_buffers.py) must never be able to change OnSRAM's simulated
numbers, and a change here must never affect COSMA. Only the underlying
scalesim/ engine itself (scale_config, topologies, layouts,
single_layer_sim, double_buffered_scratchpad) and the generic, dict-shaped
spm_common.spm_allocator.SpmAllocator (already relied on throughout this
port for physical-consistency checking, and used the identical way by
COSMA's own baseline.py -- both import the one shared module directly,
since it has no paper-specific logic at all) are shared; the actual
layer-simulation orchestration below is OnSRAM's own copy.

Only CONV2D/DEPTHWISE_CONV2D layers are actually simulated by SCALE-Sim
(see onsram_helpers/topology.py) -- every other model.json layer id (ADD,
DENSE, REDUCE_MEAN, ...) gets compute_cycles=0, compulsory_dram_bytes=0,
since SCALE-Sim's systolic array model doesn't cover them.

Each layer's ifmap/ofmap/filter SCALE-Sim SRAM buffers are sized to the
*real* tensor/weight byte sizes from model.json (via single_layer_sim.
set_memory_system()), not scale.cfg's flat IfmapSramSzkB/FilterSramSzkB/
OfmapSramSzkB defaults -- consistent with OnSRAM's own whole-tensor
residency model (onsram_helpers/pinning.py: a tensor is either pinned in
full or not at all, never partially).

Two entry points:
  - run_baseline(): the plain, OnSRAM-unaware simulation -- every layer
    independently, exactly SCALE-Sim's ordinary behavior, no cross-layer
    memory sharing. Doesn't depend on any SPM budget.
  - run_onsram_aware(): the same simulation, but driven by OnSRAM's own
    plan's resident_action (from onsram_helpers.pinning.build_resident_action())
    via OnsramResidentReadBuffer/OnsramResidentWriteBuffer
    (onsram_helpers/resident_buffers.py) -- a tensor OnSRAM's greedy
    pinning decision says is already resident is genuinely simulated as a
    zero-cost SRAM hit, and a layer's own freshly-created output is
    genuinely simulated as staying on-chip, never drained to DRAM. Every
    genuine fetch/write still runs through SCALE-Sim's exact unmodified
    logic; only the specific case OnSRAM's decision has already resolved
    is short-circuited.
"""
import json
import math
import os
from typing import Dict, List, Tuple

from scalesim.scale_config import scale_config
from scalesim.topology_utils import topologies
from scalesim.layout_utils import layouts
from scalesim.single_layer_sim import single_layer_sim
from scalesim.memory.double_buffered_scratchpad_mem import double_buffered_scratchpad

from .topology import build_onsram_topology
from .resident_buffers import OnsramResidentReadBuffer, OnsramResidentWriteBuffer

from spm_common.spm_allocator import SpmAllocator  # generic, paper-agnostic -- see module docstring

_ONSRAM_HELPERS_DIR = os.path.dirname(os.path.abspath(__file__))
_ONSRAM_DIR = os.path.dirname(_ONSRAM_HELPERS_DIR)

# Bytes per tensor element on the paper's hardware: FP16. The paper's
# accelerator has an FP16 SIMD array (Sec. 6), and OnSRAM's own reference
# implementation hard-codes 2 bytes/element too (see fom.py's "force
# float16"). The exported models are float32, but every OnSRAM size -- the
# SPM budget check (run_onsram.run_onsram() rescales its tensors with
# this), SCALE-Sim buffer sizes, and DRAM traffic -- uses this one value,
# so the 2MB SPM and the 32 B/cycle DRAM link see the paper's data sizes.
# COSMA is unaffected (its own runner keeps model.json's dtype sizes).
BYTES_PER_ELEMENT = 2


def _tensor_size_bytes(shape, dtype: str = None) -> int:
    """
    A tensor's size at the paper's precision (BYTES_PER_ELEMENT per
    element). dtype is accepted for call-site compatibility but ignored:
    model.json's float32 is the export format, not the paper's hardware.
    """
    num_elements = math.prod(shape) if shape else 1
    return num_elements * BYTES_PER_ELEMENT


def _activation_input_tensor_id(layer: dict):
    """
    Returns the tensor id of the layer's activation input
    Returns None if the layer has no inputs at all.
    """
    inputs = layer.get('inputs', [])
    if not inputs:
        return None
    inputs_from = layer.get('inputs_from', [-1] * len(inputs))
    act_idx = next((i for i, src in enumerate(inputs_from) if src != -1), 0)
    return inputs[act_idx]


def _ofmap_tensor_id(layer: dict):
    """
    The tensor id of this layer's own freshly-produced output -- mirrors
    _activation_input_tensor_id()'s role for inputs. Only looks at
    outputs[0] -- the same simplification _layer_operand_bytes()'s own
    ofmap_bytes already makes. Returns None if the layer has no outputs.
    """
    outputs = layer.get('outputs', [])
    return outputs[0] if outputs else None


def _split_free_room(room: int, needs: dict) -> dict:
    """
    Splits a layer's free SPM room (what's left after resident tensors)
    among its working buffers the way a tiled accelerator shares it: every
    operand gets a part. Equal shares, except an operand that needs less
    than its share keeps only what it needs and the rest is re-split among
    the others (max-min fair / water-filling). So if everything fits,
    everyone gets their full size; a small operand (e.g. a 2.5KB filter)
    always fits whole; and big operands share the remainder and stream
    through it in tiles. needs: {name: bytes}; returns {name: bytes}, which
    sums to at most room.
    """
    shares = {}
    left = room
    pending = sorted(needs, key=lambda n: needs[n])
    while pending:
        fair = left // len(pending)
        name = pending.pop(0)
        shares[name] = min(needs[name], fair)
        left -= shares[name]
    return shares


def _print_share_events(tag: str, operand: str, events: list, num_timesteps: int) -> None:
    """One summary line for how often an operand didn't fit its share of the free room."""
    if events:
        worst_t, worst_lid, worst_short = max(events, key=lambda e: e[2])
        print(f"[{tag} SPM] {operand} bigger than its share of the free room on "
              f"{len(events)} of {num_timesteps} timestep(s) (worst case {worst_short} "
              f"bytes short at t={worst_t}, layer {worst_lid}) -- logged only: under "
              f"ideal tiling it streams through its share, each element read once")
    else:
        print(f"[{tag} SPM] {operand} fits its share of the free room on all "
              f"{num_timesteps} timesteps")


def _ifmap_in_spm(resident_action: dict, handoff_action: dict, ifmap_id, t: int) -> bool:
    """
    Whether this layer reads its activation input from the SPM at t:
    either still resident ('P' in resident_action), or an Overwrite
    Optimization hand-off read ('H' in handoff_action -- the tensor is
    read from the SPM while this layer's own output takes over its
    space; see pinning.build_handoff_action()).
    """
    if resident_action is None or ifmap_id is None:
        return False
    return (resident_action.get((ifmap_id, t)) == 'P'
            or (handoff_action or {}).get((ifmap_id, t)) == 'H')


def _ofmap_pinned(resident_action: dict, layer: dict, t: int) -> bool:
    """
    Whether this layer's output stays on-chip: only when OnSRAM pinned it,
    i.e. it's created in the SPM ('C') at this layer's own t. Every other
    output is written back to DRAM (paper Sec. 3.1/3.2).
    """
    ofmap_id = _ofmap_tensor_id(layer)
    return (resident_action is not None and ofmap_id is not None
            and resident_action.get((ofmap_id, t)) == 'C')


def _layer_operand_bytes(layer: dict, tensor_shapes: Dict[int, dict]):
    """
    Returns (ifmap_bytes, ofmap_bytes, filter_bytes) for one model.json
    layer, from the real tensor shapes / weight+bias sizes -- not a
    config-driven guess.
    """
    ifmap_id = _activation_input_tensor_id(layer)
    ifmap_bytes = 0
    if ifmap_id is not None:
        t = tensor_shapes[ifmap_id]
        ifmap_bytes = _tensor_size_bytes(t['shape'], t['dtype'])

    outputs = layer.get('outputs', [])
    ofmap_bytes = 0
    if outputs:
        t = tensor_shapes[outputs[0]]
        ofmap_bytes = _tensor_size_bytes(t['shape'], t['dtype'])

    # Weights/bias at the paper's precision too (model.json stores their
    # float32/int32 byte size plus an element count).
    filter_elems = sum(part.get('elements', part.get('size', 0) // 4)
                       for part in (layer.get('weights', {}), layer.get('bias', {})))
    filter_bytes = filter_elems * BYTES_PER_ELEMENT

    return ifmap_bytes, ofmap_bytes, filter_bytes


def _make_memory_system(config, topo, layer_id: int,
                         ifmap_buf_size_bytes: int, filter_buf_size_bytes: int,
                         ofmap_buf_size_bytes: int, verbose: bool,
                         ifmap_resident: bool = False,
                         ofmap_stays_on_chip: bool = False):
    """
    Builds a double_buffered_scratchpad sized to the real operand byte
    counts for this layer, following the same pattern single_layer_sim.run()
    uses internally for its own default memory system -- just with real
    sizes instead of scale.cfg's flat per-buffer defaults.

    ifmap_resident/ofmap_stays_on_chip: when True, install
    OnsramResidentReadBuffer/OnsramResidentWriteBuffer instead of
    SCALE-Sim's default buffer classes, so this layer's ifmap fetch /
    ofmap drain is genuinely simulated as free -- see run_onsram_aware()'s
    docstring for when each applies.
    """
    mem = double_buffered_scratchpad()
    if ofmap_stays_on_chip:
        mem.ofmap_buf = OnsramResidentWriteBuffer()
        mem.ofmap_buf.stays_on_chip = True

    if config.use_user_dram_bandwidth():
        bw_list = config.get_bandwidths_as_list()
        ifmap_bw = getattr(config, 'ifmap_sram_bank_bandwidth', 10)
        filter_bw = getattr(config, 'filter_sram_bank_bandwidth', 10)
        ofmap_bw = bw_list[0]
        estimate_bandwidth_mode = False
    else:
        _, arr_col = config.get_array_dims()
        ifmap_bw = filter_bw = 10
        ofmap_bw = arr_col
        estimate_bandwidth_mode = True

    mem.set_params(
        layer_id=layer_id,
        word_size=1,
        ifmap_buf_size_bytes=max(ifmap_buf_size_bytes, 1),
        filter_buf_size_bytes=max(filter_buf_size_bytes, 1),
        ofmap_buf_size_bytes=max(ofmap_buf_size_bytes, 1),
        rd_buf_active_frac=0.5,
        wr_buf_active_frac=0.5,
        ifmap_backing_buf_bw=ifmap_bw,
        filter_backing_buf_bw=filter_bw,
        ofmap_backing_buf_bw=ofmap_bw,
        verbose=verbose,
        estimate_bandwidth_mode=estimate_bandwidth_mode,
        ifmap_sram_bank_num=getattr(config, 'ifmap_sram_bank_num', 1),
        ifmap_sram_bank_port=getattr(config, 'ifmap_sram_bank_port', 2),
        filter_sram_bank_num=getattr(config, 'filter_sram_bank_num', 1),
        filter_sram_bank_port=getattr(config, 'filter_sram_bank_port', 2),
        config=config,
        topo=topo,
        ifmap_buf_class=OnsramResidentReadBuffer if ifmap_resident else None,
    )
    if ifmap_resident:
        mem.ifmap_buf.fully_resident = True
    return mem


def _simulate_layer(config, topo, layout, row: int, layer: dict, t: int,
                     tensor_shapes: Dict[int, dict], verbose: bool,
                     resident_action: dict = None,
                     handoff_action: dict = None) -> dict:
    """
    Runs one conv-like layer and returns its {compute_cycles,
    ifmap_dram_bytes, filter_dram_bytes, ofmap_dram_bytes}.

    t: the abstract schedule timestep this layer runs at -- resident_action/
    spm_plan are keyed by (tensor_id, t) from
    onsram_helpers.pinning.decide_pinning()/build_resident_action(), NOT by
    layer id; the two coincide for OnSRAM's schedule too (schedule is
    [(t, layer_id), ...] built directly from the BFS-DFS order), so this
    must be the caller's real schedule position, matching _run_layers()'s
    `schedule` param.

    resident_action: None for the plain, OnSRAM-unaware baseline (every
    fetch/drain simulated normally). When given (a
    {(tensor_id, t): 'C'|'P'} map -- OnSRAM never spills or retrieves, see
    onsram_helpers/pinning.py), this layer's ifmap read is installed as a
    genuine zero-cost hit when the activation input is resident via 'P'
    (OnSRAM says it's already on-chip -- no real event to simulate), and
    the ofmap write stays on-chip only when OnSRAM pinned this layer's
    output ('C' at t); an unpinned output is written back to DRAM as
    usual. That's the paper's rule (Sec. 3.2: an activation "can be
    pinned in the on-chip SPM as opposed to a write-back to external
    memory"). This is what run_onsram_aware() uses; run_baseline() always
    passes None.

    handoff_action: OnSRAM's {(tensor_id, t): 'H'} Overwrite Optimization
    hand-off reads (pinning.build_handoff_action()). An 'H' ifmap is read
    from the SPM, exactly like a 'P' one. None/empty for run_baseline().
    """
    ifmap_bytes, ofmap_bytes, filter_bytes = _layer_operand_bytes(layer, tensor_shapes)

    ifmap_resident = _ifmap_in_spm(resident_action, handoff_action,
                                    _activation_input_tensor_id(layer), t)
    ofmap_pinned = _ofmap_pinned(resident_action, layer, t)

    # Buffers are always their natural size (ideal tiling -- see
    # _run_layers()'s comment on the free-room split).
    mem_sys = _make_memory_system(
        config, topo, row, ifmap_bytes, filter_bytes, ofmap_bytes, verbose,
        ifmap_resident=ifmap_resident,
        ofmap_stays_on_chip=ofmap_pinned,
    )

    sim = single_layer_sim()
    sim.set_params(layer_id=row, config_obj=config, topology_obj=topo,
                    layout_obj=layout, verbose=verbose)
    sim.set_memory_system(mem_sys)
    sim.run()

    # Timing follows the paper's own model (Sec. 6): per layer,
    # max(compute_time, data_xfer_time) -- the caller takes that max -- with
    # "ideal tiling ... each data element is fetched once" and no stalls.
    # So from SCALE-Sim we take only pure compute time (total cycles minus
    # its own memory-stall cycles, which the data-transfer term already
    # covers), and DRAM traffic is each real tensor once, at the paper's
    # precision: the input unless it's in the SPM ('P' or 'H'), the output
    # unless OnSRAM pinned it, the weights always. SCALE-Sim's own DRAM
    # counts include re-reads the paper's baseline doesn't have (measured
    # 1.3-4.1x the fetch-once traffic on MobileNet/MobileNetV2/SqueezeNet),
    # which pushed speedups above the paper's own bounds.
    compute_items = sim.get_compute_report_items()
    total_cycles = compute_items[1]  # index 1 = Total Cycles (excl. prefetch)
    stall_cycles = compute_items[2]
    return {
        'compute_cycles': int(total_cycles - stall_cycles),
        # kept split out (rather than pre-summed) so callers can attribute
        # each component separately.
        'ifmap_dram_bytes': 0 if ifmap_resident else ifmap_bytes,
        'filter_dram_bytes': filter_bytes,
        'ofmap_dram_bytes': 0 if ofmap_pinned else ofmap_bytes,
    }


def _run_layers(model_json_path: str, config_path: str,
                 topology_csv_path: str, verbose: bool,
                 resident_action: dict = None, spm_plan: dict = None,
                 tensors: dict = None, memory_budget_bytes: int = None,
                 schedule: List[Tuple[int, int]] = None,
                 handoff_action: dict = None) -> Dict[int, dict]:
    """
    schedule: optional [(t, layer_id), ...] execution order, sorted by t.
    None (the default -- always what run_baseline() uses) falls back to
    model.json's own layer order, t == position == layer id.
    """
    if resident_action is not None:
        assert spm_plan is not None and tensors is not None and memory_budget_bytes is not None, (
            "run_onsram_aware()'s plan (resident_action/spm_plan/tensors/"
            "memory_budget_bytes) must be supplied together")

    if topology_csv_path is None:
        topology_csv_path = os.path.join(_ONSRAM_DIR, 'topology.csv')

    layer_id_to_row = build_onsram_topology(model_json_path, topology_csv_path)

    config = scale_config()
    config.read_conf_file(config_path)

    topo = topologies()
    topo.load_arrays(topofile=topology_csv_path, mnk_inputs=False)

    # Not using SCALE-Sim's custom-layout mode (scale.cfg has IfmapCustomLayout/
    # FilterCustomLayout = False), so the layout object is left unloaded --
    # single_layer_sim only touches it when those flags are set.
    layout = layouts()

    with open(model_json_path, 'r') as f:
        model = json.load(f)
    tensor_shapes: Dict[int, dict] = {t['id']: t for t in model['tensors']}
    layer_by_id: Dict[int, dict] = {layer['id']: layer for layer in model['layers']}

    if schedule is None:
        # Today's behavior: model.json's own order, t == position == layer id.
        schedule = [(i, layer['id']) for i, layer in enumerate(model['layers'])]

    allocator = None
    if resident_action is not None:
        allocator = SpmAllocator(tensors, spm_plan, resident_action, memory_budget_bytes)

    # Budget-vs-working-set check: SpmAllocator (above) verifies resident
    # tensors fit memory_budget_bytes against each other; they keep their
    # space from one layer to the next. The room they leave
    # (allocator.remaining_budget_for(), spm_common's) is each layer's
    # working area, split among ifmap, filter and ofmap the way a tiled
    # accelerator shares it (_split_free_room(): equal shares, an operand
    # that needs less keeps just what it needs). An ifmap already in the
    # SPM needs no working room.
    #
    # The split is logged only, never fed into SCALE-Sim's buffer sizes.
    # This is the paper's own "ideal tiling" assumption (OnSRAM Sec. 6:
    # "each data element is fetched once"): an operand bigger than its
    # share streams through it tile by tile, each element read once.
    # Shrinking SCALE-Sim's read buffers to the share instead was tried and
    # measured to produce thrashing no real accelerator has (ResNet-50
    # layer 61: 118M weight reads for 2.36M weights, and a smaller buffer
    # giving less traffic than a bigger one). OnSRAM's pinning's own decisions are
    # untouched and never see any of this.
    #
    # budget_overflow_events is kept as a frozen historical benchmark: it
    # compares the layer's full filter bytes against what's resident, i.e.
    # "would this layer's whole weight tensor fit next to the residents."
    budget_overflow_events = []
    ifmap_ceiling_events = []
    handoff_reads = 0
    pinned_outputs = 0
    simulated_layers = 0
    ofmap_ceiling_events = []
    filter_ceiling_events = []

    row_to_stats: Dict[int, dict] = {}
    for t, lid in schedule:
        layer = layer_by_id[lid]
        if allocator is not None:
            # Replay every timestep's C/P transitions, including non-conv
            # ones (ADD, DENSE, ...) below -- a tensor can be legitimately
            # resident through a non-conv timestep too, and skipping those
            # would silently corrupt the allocator's live state. Raises
            # SpmAllocationError loudly if the plan is ever physically
            # inconsistent. Called with t (the abstract schedule timestep
            # resident_action/spm_plan are keyed by), not lid.
            allocator.step(t)
        if lid not in layer_id_to_row:
            continue

        if allocator is not None:
            ifmap_bytes, ofmap_bytes, filter_bytes = _layer_operand_bytes(layer, tensor_shapes)
            combined = allocator.occupied_bytes() + filter_bytes
            if combined > memory_budget_bytes:
                budget_overflow_events.append((t, lid, combined - memory_budget_bytes))

            ifmap_id = _activation_input_tensor_id(layer)
            ofmap_id = _ofmap_tensor_id(layer)
            claim_ids = tuple(tid for tid in (ifmap_id, ofmap_id) if tid is not None)
            room_all = allocator.remaining_budget_for(claim_ids, t)

            # An ifmap already in the SPM (resident/hand-off) needs no
            # working room; everything else shares room_all fairly.
            simulated_layers += 1
            pinned_outputs += _ofmap_pinned(resident_action, layer, t)
            ifmap_in_spm = ifmap_id is not None and _ifmap_in_spm(resident_action, handoff_action, ifmap_id, t)
            if ifmap_id is not None and (handoff_action or {}).get((ifmap_id, t)) == 'H':
                handoff_reads += 1
            needs = {'filter': filter_bytes}
            if ifmap_id is not None:
                needs['ifmap'] = 0 if ifmap_in_spm else ifmap_bytes
            if ofmap_id is not None:
                needs['ofmap'] = ofmap_bytes
            shares = _split_free_room(room_all, needs)
            assert sum(shares.values()) <= room_all, (
                f"free-room split arithmetic bug at t={t}, layer {lid}")

            if shares['filter'] < filter_bytes:
                filter_ceiling_events.append((t, lid, filter_bytes - shares['filter']))
            if ifmap_id is not None:
                if not ifmap_in_spm and shares['ifmap'] < ifmap_bytes:
                    ifmap_ceiling_events.append((t, lid, ifmap_bytes - shares['ifmap']))
            if ofmap_id is not None:
                if shares['ofmap'] < ofmap_bytes:
                    ofmap_ceiling_events.append((t, lid, ofmap_bytes - shares['ofmap']))

        row_to_stats[layer_id_to_row[lid]] = _simulate_layer(
            config, topo, layout, layer_id_to_row[lid], layer, t, tensor_shapes,
            verbose, resident_action=resident_action,
            handoff_action=handoff_action)

    if allocator is not None:
        # Deliberately unconditional, not gated on `verbose`, that flag
        # also enables SCALE-Sim's own internal per-layer tqdm progress
        # bars, which would flood the terminal with dozens of bars just to
        # surface this one cheap, already-computed summary line.
        print(f"[OnSRAM SPM] verified {allocator.steps_taken()} timesteps, "
              f"peak occupancy {allocator.peak_occupied_bytes()}/{memory_budget_bytes} "
              f"bytes, 0 violations")
        print(f"[OnSRAM SPM] {handoff_reads} of {len(handoff_action or {})} Overwrite "
              f"Optimization hand-off read(s) ('H') served from SPM by a simulated "
              f"conv layer (the rest are read by non-conv layers SCALE-Sim doesn't run)")
        print(f"[OnSRAM SPM] {pinned_outputs} of {simulated_layers} simulated layers keep "
              f"their output on-chip (pinned); the other {simulated_layers - pinned_outputs} "
              f"write it back to DRAM")
        if budget_overflow_events:
            worst_t, worst_lid, worst_over = max(budget_overflow_events, key=lambda e: e[2])
            print(f"[OnSRAM SPM] WARNING: {len(budget_overflow_events)} of {len(schedule)} "
                  f"timestep(s) would exceed the {memory_budget_bytes}-byte budget under "
                  f"the OLD, fully-unconstrained filter/weight treatment (layer's full "
                  f"filter bytes on top of what's resident; see the filter line "
                  f"below); worst "
                  f"case {worst_over} bytes over at t={worst_t} (layer {worst_lid})")
        else:
            print(f"[OnSRAM SPM] budget-vs-working-set check: 0 of {len(schedule)} "
                  f"timesteps exceed the {memory_budget_bytes}-byte budget")
        _print_share_events('OnSRAM', 'ifmap', ifmap_ceiling_events, len(schedule))
        _print_share_events('OnSRAM', 'ofmap', ofmap_ceiling_events, len(schedule))
        _print_share_events('OnSRAM', 'filter', filter_ceiling_events, len(schedule))

    layer_stats: Dict[int, dict] = {}
    for layer in model['layers']:
        lid = layer['id']
        if lid in layer_id_to_row:
            layer_stats[lid] = row_to_stats[layer_id_to_row[lid]]
        else:
            layer_stats[lid] = {'compute_cycles': 0, 'ifmap_dram_bytes': 0,
                                 'filter_dram_bytes': 0, 'ofmap_dram_bytes': 0}

    return layer_stats


def run_baseline(model_json_path: str, config_path: str,
                  topology_csv_path: str = None,
                  verbose: bool = False) -> Dict[int, dict]:
    """
    Returns layer_stats: model.json layer id -> {compute_cycles,
    ifmap_dram_bytes, filter_dram_bytes, ofmap_dram_bytes} for every layer
    in model.json (conv-like layers get real SCALE-Sim numbers, everything
    else gets zeros). This is the plain, OnSRAM-unaware simulation -- every
    layer independently, no cross-layer memory sharing at all (SCALE-Sim's
    ordinary behavior). Does not depend on any SPM budget.
    """
    return _run_layers(model_json_path, config_path, topology_csv_path, verbose)


def run_onsram_aware(model_json_path: str, config_path: str, resident_action: dict,
                      spm_plan: dict, tensors: dict, memory_budget_bytes: int,
                      topology_csv_path: str = None,
                      verbose: bool = False,
                      schedule: List[Tuple[int, int]] = None,
                      handoff_action: dict = None) -> Dict[int, dict]:
    """
    Same per-layer numbers as run_baseline(), except driven by OnSRAM's
    actual plan (resident_action, from
    onsram_helpers.pinning.build_resident_action()): a layer's ifmap fetch
    is genuinely simulated as free when its activation input is resident
    via 'P' (or read at an 'H' hand-off), and a layer's ofmap write stays
    on-chip only when OnSRAM pinned that output -- see _simulate_layer()'s
    docstring. Unlike run_baseline(), this does depend on the SPM budget
    (indirectly, via which resident_action was decided for) and must be
    re-run per budget.

    schedule: [(t, layer_id), ...] -- OnSRAM's own BFS-DFS execution
    order, always required here (unlike COSMA's equivalent, OnSRAM always
    reorders layers relative to model.json's own order, so there's no
    meaningful None-defaults-to-identity-order shortcut for it).

    spm_plan/tensors/memory_budget_bytes drive a live SpmAllocator replay
    alongside the real SCALE-Sim simulation -- an independent,
    physically-checked verification that resident_action's claims are
    actually realizable at this budget, not just trusted. Required (no
    default) so this doesn't silently keep running without that check.

    handoff_action: {(tensor_id, t): 'H'} from
    pinning.build_handoff_action() -- Overwrite Optimization hand-off
    reads, simulated as SPM hits. Deliberately separate from
    resident_action, which the shared SpmAllocator replays.
    """
    return _run_layers(model_json_path, config_path, topology_csv_path, verbose,
                        resident_action=resident_action, spm_plan=spm_plan,
                        tensors=tensors, memory_budget_bytes=memory_budget_bytes,
                        schedule=schedule, handoff_action=handoff_action)
