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

_DTYPE_BYTES = {
    'float32': 4, 'float16': 2, 'bfloat16': 2,
    'int32': 4, 'int8': 1, 'uint8': 1,
}


def _tensor_size_bytes(shape, dtype: str) -> int:
    """
    Own copy of graph_builder.compute_size_bytes() -- same dtype table,
    same formula. Duplicated rather than imported for the same reason as
    everything else in this module (see module docstring): this file's
    only cross-project dependency is the shared scalesim/ engine plus
    SpmAllocator, nothing under cosma/.
    """
    bytes_per_elem = _DTYPE_BYTES.get(dtype.lower(), 4)
    num_elements = math.prod(shape) if shape else 1
    return num_elements * bytes_per_elem


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

    filter_bytes = (layer.get('weights', {}).get('size', 0)
                     + layer.get('bias', {}).get('size', 0))

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
                     resident_action: dict = None) -> dict:
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
    the ofmap write is always installed as staying on-chip (a layer's own
    freshly-created output never needs a DRAM round-trip in this model --
    see resident_buffers.py's docstring). This is what run_onsram_aware()
    uses; run_baseline() always passes None.
    """
    ifmap_bytes, ofmap_bytes, filter_bytes = _layer_operand_bytes(layer, tensor_shapes)

    ifmap_resident = False
    if resident_action is not None:
        ifmap_id = _activation_input_tensor_id(layer)
        if ifmap_id is not None:
            ifmap_resident = resident_action.get((ifmap_id, t)) == 'P'

    mem_sys = _make_memory_system(
        config, topo, row, ifmap_bytes, filter_bytes, ofmap_bytes, verbose,
        ifmap_resident=ifmap_resident,
        ofmap_stays_on_chip=resident_action is not None,
    )

    sim = single_layer_sim()
    sim.set_params(layer_id=row, config_obj=config, topology_obj=topo,
                    layout_obj=layout, verbose=verbose)
    sim.set_memory_system(mem_sys)
    sim.run()

    compute_items = sim.get_compute_report_items()
    total_cycles = compute_items[1]  # index 1 = Total Cycles (excl. prefetch)

    detail_items = sim.get_detail_report_items()
    return {
        'compute_cycles': int(total_cycles),
        # kept split out (rather than pre-summed) so callers can attribute
        # each component separately.
        'ifmap_dram_bytes': int(detail_items[11]),
        'filter_dram_bytes': int(detail_items[14]),
        'ofmap_dram_bytes': int(detail_items[17]),
    }


def _run_layers(model_json_path: str, config_path: str,
                 topology_csv_path: str, verbose: bool,
                 resident_action: dict = None, spm_plan: dict = None,
                 tensors: dict = None, memory_budget_bytes: int = None,
                 schedule: List[Tuple[int, int]] = None) -> Dict[int, dict]:
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
        row_to_stats[layer_id_to_row[lid]] = _simulate_layer(
            config, topo, layout, layer_id_to_row[lid], layer, t, tensor_shapes,
            verbose, resident_action=resident_action)

    if allocator is not None:
        # Deliberately unconditional, not gated on `verbose`, that flag
        # also enables SCALE-Sim's own internal per-layer tqdm progress
        # bars, which would flood the terminal with dozens of bars just to
        # surface this one cheap, already-computed summary line.
        print(f"[OnSRAM SPM] verified {allocator.steps_taken()} timesteps, "
              f"peak occupancy {allocator.peak_occupied_bytes()}/{memory_budget_bytes} "
              f"bytes, 0 violations")

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
                      schedule: List[Tuple[int, int]] = None) -> Dict[int, dict]:
    """
    Same per-layer numbers as run_baseline(), except driven by OnSRAM's
    actual plan (resident_action, from
    onsram_helpers.pinning.build_resident_action()): a layer's ifmap fetch
    is genuinely simulated as free when its activation input is resident
    via 'P', and every layer's ofmap write is genuinely simulated as
    staying on-chip (never drained to DRAM) -- see _simulate_layer()'s
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
    """
    return _run_layers(model_json_path, config_path, topology_csv_path, verbose,
                        resident_action=resident_action, spm_plan=spm_plan,
                        tensors=tensors, memory_budget_bytes=memory_budget_bytes,
                        schedule=schedule)
