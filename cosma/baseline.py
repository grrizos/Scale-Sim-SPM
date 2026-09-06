# cosma/baseline.py
"""
Drives SCALE-Sim in-process (no subprocess, no intermediate CSV round-trip)
to get per-layer compute cycles and compulsory DRAM bytes for every layer
in model.json.

Only CONV2D/DEPTHWISE_CONV2D layers are actually simulated by SCALE-Sim
(see topology_builder.py) -- every other model.json layer id (ADD, DENSE,
REDUCE_MEAN, ...) gets compute_cycles=0, compulsory_dram_bytes=0, since
SCALE-Sim's systolic array model doesn't cover them.

Each layer's ifmap/ofmap/filter SCALE-Sim SRAM buffers are sized to the
*real* tensor/weight byte sizes from model.json (via single_layer_sim.
set_memory_system(), the same officially-supported hook used by the old
smm_scalesim_runner.py), not scale.cfg's flat IfmapSramSzkB/FilterSramSzkB/
OfmapSramSzkB defaults. This matters for consistency with cosma_Ilp.py:
COSMA's ILP already assumes a resident tensor occupies its full size with
no partial/thrashed residency (Eq.9), so SCALE-Sim's own buffer sizing
should reflect that same "whole tensor, no internal tiling" assumption
rather than an arbitrary fixed size unrelated to the actual data -- a
64KB default buffer for a 588KB tensor or a 2KB tensor were previously
both simulated identically, which made SCALE-Sim's own compute/mapping-
efficiency numbers disconnected from the sizes COSMA's ILP is reasoning
about.

Two entry points:
  - run_baseline(): the plain, COSMA-unaware simulation -- every layer
    independently, exactly SCALE-Sim's ordinary behavior, no cross-layer
    memory sharing. Doesn't depend on any SPM budget.
  - run_cosma_aware(): the same simulation, but driven by a COSMA plan's
    resident_action (from cosma_Ilp.extract_results()) via
    CosmaResidentReadBuffer/CosmaResidentWriteBuffer
    (scalesim/memory/cosma_resident_buffers.py) -- a tensor COSMA's ILP
    says is already resident is genuinely simulated as a zero-cost SRAM
    hit, and a layer's own freshly-created output is genuinely simulated
    as staying on-chip, never drained to DRAM (see those classes'
    docstrings for exactly why, and cosma/ITERATION_HISTORY.md for the
    investigation that grounded this design). This replaces what used to
    be an after-the-fact analytic adjustment in run_cosma.py with real,
    engine-driven numbers -- every genuine fetch/write still runs through
    SCALE-Sim's exact unmodified logic; only the specific case COSMA's
    ILP has already resolved is short-circuited.
"""
import json
import os
from typing import Dict

from scalesim.scale_config import scale_config
from scalesim.topology_utils import topologies
from scalesim.layout_utils import layouts
from scalesim.single_layer_sim import single_layer_sim
from scalesim.memory.double_buffered_scratchpad_mem import double_buffered_scratchpad
from scalesim.memory.cosma_resident_buffers import (
    CosmaResidentReadBuffer, CosmaResidentWriteBuffer)

from topology_builder import build_topology
from graph_builder import compute_size_bytes

HERE = os.path.dirname(os.path.abspath(__file__))


def _activation_input_tensor_id(layer: dict):
    """
    The tensor id of this layer's activation input (as opposed to
    weight/bias/network-input, all marked inputs_from==-1): prefers the
    first input actually produced by another layer; layer 0 has none, so
    falls back to inputs[0] (the network's raw input tensor) -- see
    graph_builder.py's module docstring for why inputs_from==-1 doesn't
    always mean "not real data". Returns None if the layer has no inputs
    at all.
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
        ifmap_bytes = compute_size_bytes(t['shape'], t['dtype'])

    outputs = layer.get('outputs', [])
    ofmap_bytes = 0
    if outputs:
        t = tensor_shapes[outputs[0]]
        ofmap_bytes = compute_size_bytes(t['shape'], t['dtype'])

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
    uses internally for its own default memory system (see
    scalesim/single_layer_sim.py) -- just with real sizes instead of
    scale.cfg's flat per-buffer defaults.

    ifmap_resident/ofmap_stays_on_chip: when True, install
    CosmaResidentReadBuffer/CosmaResidentWriteBuffer (scalesim/memory/
    cosma_resident_buffers.py) instead of SCALE-Sim's default buffer
    classes, so this layer's ifmap fetch / ofmap drain is genuinely
    simulated as free -- see run_cosma_aware()'s docstring for when each
    applies.
    """
    mem = double_buffered_scratchpad()
    if ofmap_stays_on_chip:
        mem.ofmap_buf = CosmaResidentWriteBuffer()
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
        ifmap_buf_class=CosmaResidentReadBuffer if ifmap_resident else None,
    )
    if ifmap_resident:
        mem.ifmap_buf.fully_resident = True
    return mem


def _simulate_layer(config, topo, layout, row: int, layer: dict,
                     tensor_shapes: Dict[int, dict], verbose: bool,
                     resident_action: dict = None) -> dict:
    """
    Runs one conv-like layer and returns its {compute_cycles,
    ifmap_dram_bytes, filter_dram_bytes, ofmap_dram_bytes}.

    resident_action: None for the plain, COSMA-unaware baseline (every
    fetch/drain simulated normally). When given (a
    {(tensor_id, layer_id): 'C'|'P'|'R'|'S'} map from
    cosma_Ilp.extract_results()), this layer's ifmap read is installed as
    a genuine zero-cost hit when the activation input is resident via 'P'
    (COSMA says it's already on-chip -- no real event to simulate), and
    the ofmap write is always installed as staying on-chip (a layer's own
    freshly-created output never needs a DRAM round-trip in COSMA's
    model -- see cosma_resident_buffers.py's docstring on Eq.3). This is
    what run_cosma_aware() uses; run_baseline() always passes None.
    """
    ifmap_bytes, ofmap_bytes, filter_bytes = _layer_operand_bytes(layer, tensor_shapes)

    ifmap_resident = False
    if resident_action is not None:
        ifmap_id = _activation_input_tensor_id(layer)
        if ifmap_id is not None:
            ifmap_resident = resident_action.get((ifmap_id, layer['id'])) == 'P'

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
        # each component separately -- see run_cosma.py's module docstring.
        'ifmap_dram_bytes': int(detail_items[11]),
        'filter_dram_bytes': int(detail_items[14]),
        'ofmap_dram_bytes': int(detail_items[17]),
    }


def _run_layers(model_json_path: str, config_path: str,
                 topology_csv_path: str, verbose: bool,
                 resident_action: dict = None) -> Dict[int, dict]:
    if topology_csv_path is None:
        topology_csv_path = os.path.join(HERE, 'topology.csv')

    layer_id_to_row = build_topology(model_json_path, topology_csv_path)

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

    row_to_stats: Dict[int, dict] = {}
    for layer in model['layers']:
        lid = layer['id']
        if lid not in layer_id_to_row:
            continue
        row_to_stats[layer_id_to_row[lid]] = _simulate_layer(
            config, topo, layout, layer_id_to_row[lid], layer, tensor_shapes,
            verbose, resident_action=resident_action)

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
    else gets zeros). This is the plain, COSMA-unaware simulation -- every
    layer independently, no cross-layer memory sharing at all (SCALE-Sim's
    ordinary behavior). Does not depend on any SPM budget.
    """
    return _run_layers(model_json_path, config_path, topology_csv_path, verbose)


def run_cosma_aware(model_json_path: str, config_path: str, resident_action: dict,
                     topology_csv_path: str = None,
                     verbose: bool = False) -> Dict[int, dict]:
    """
    Same per-layer numbers as run_baseline(), except driven by COSMA's
    actual plan (resident_action, from cosma_Ilp.extract_results()): a
    layer's ifmap fetch is genuinely simulated as free when its
    activation input is resident via 'P', and every layer's ofmap write
    is genuinely simulated as staying on-chip (never drained to DRAM) --
    see _simulate_layer()'s docstring. Unlike run_baseline(), this does
    depend on the SPM budget (indirectly, via which resident_action was
    solved for) and must be re-run per budget.
    """
    return _run_layers(model_json_path, config_path, topology_csv_path, verbose,
                        resident_action=resident_action)


if __name__ == '__main__':
    stats = run_baseline(
        model_json_path=os.path.join(HERE, 'model.json'),
        config_path=os.path.join(os.path.dirname(HERE), 'configs', 'scale.cfg'),
    )
    total_cycles = sum(s['compute_cycles'] for s in stats.values())
    total_dram = sum(s['ifmap_dram_bytes'] + s['filter_dram_bytes'] + s['ofmap_dram_bytes']
                      for s in stats.values())
    print(f"Layers: {len(stats)}")
    print(f"Total compute cycles (sum, no overlap): {total_cycles}")
    print(f"Total compulsory DRAM bytes: {total_dram}")
    nonzero = [lid for lid, s in stats.items() if s['compute_cycles'] > 0]
    print(f"Layers with nonzero compute cycles: {len(nonzero)}")
