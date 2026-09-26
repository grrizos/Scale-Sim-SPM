# cosma/helpers/baseline.py
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
import math
import os
from typing import Dict, List, Tuple

from scalesim.scale_config import scale_config
from scalesim.topology_utils import topologies
from scalesim.layout_utils import layouts
from scalesim.single_layer_sim import single_layer_sim
from scalesim.memory.double_buffered_scratchpad_mem import double_buffered_scratchpad
from scalesim.memory.cosma_resident_buffers import (
    CosmaResidentReadBuffer, CosmaResidentWriteBuffer)

from .topology_builder import build_topology
from spm_common.graph_builder import compute_size_bytes
from spm_common.spm_allocator import SpmAllocator

# cosma/ (one level up from this file's own helpers/ directory) -- kept
# pointing there, not at helpers/, so the default paths below (and every
# existing caller relying on them) are unaffected by which subdirectory
# this module physically lives in.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def _ofmap_tensor_id(layer: dict):
    """
    The tensor id of this layer's own freshly-produced output -- mirrors
    _activation_input_tensor_id()'s role for inputs. Only looks at
    outputs[0] -- the same simplification _layer_operand_bytes()'s own
    ofmap_bytes already makes. Returns None if the layer has no outputs.
    """
    outputs = layer.get('outputs', [])
    return outputs[0] if outputs else None


def _depthwise_channels(layer: dict):
    """
    C for a DEPTHWISE_CONV2D layer, else None. The topology builder writes
    a depthwise layer as Channels = 1, Num Filter = C (channels-across-
    columns), so SCALE-Sim simulates one input plane shared by all C
    columns; _simulate_layer() uses C to correct the input side.
    """
    if layer.get('op') != 'DEPTHWISE_CONV2D':
        return None
    return layer['input_shape'][3]


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


def _simulate_layer(config, topo, layout, row: int, layer: dict, t: int,
                     tensor_shapes: Dict[int, dict], verbose: bool,
                     resident_action: dict = None) -> dict:
    """
    Runs one conv-like layer and returns its {compute_cycles,
    ifmap_dram_bytes, filter_dram_bytes, ofmap_dram_bytes}.

    t: the abstract schedule timestep this layer runs at -- resident_action/
    spm_plan are keyed by (tensor_id, t) from cosma_Ilp.extract_results(),
    NOT by layer id; the two only coincide under a fixed schedule. Under
    free_schedule=True they can differ, so this must be the caller's real
    schedule position, not layer['id'] (see _run_layers()'s `schedule`
    param).

    resident_action: None for the plain, COSMA-unaware baseline (every
    fetch/drain simulated normally). When given (a
    {(tensor_id, t): 'C'|'P'|'R'|'S'} map from cosma_Ilp.extract_results()),
    this layer's ifmap read is installed as a genuine zero-cost hit when
    the activation input is resident via 'P' (COSMA says it's already
    on-chip -- no real event to simulate), and the ofmap write is always
    installed as staying on-chip (a layer's own freshly-created output
    never needs a DRAM round-trip in COSMA's model -- see
    cosma_resident_buffers.py's docstring on Eq.3). This is what
    run_cosma_aware() uses; run_baseline() always passes None.
    """
    ifmap_bytes, ofmap_bytes, filter_bytes = _layer_operand_bytes(layer, tensor_shapes)

    ifmap_resident = False
    if resident_action is not None:
        ifmap_id = _activation_input_tensor_id(layer)
        if ifmap_id is not None:
            ifmap_resident = resident_action.get((ifmap_id, t)) == 'P'

    # Buffers are always their natural size (ideal tiling -- see
    # _run_layers()'s comment on the free-room split).
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
    ifmap_dram = int(detail_items[11])
    if _depthwise_channels(layer):
        # Depthwise: SCALE-Sim simulated ONE input plane shared by all C
        # columns (see topology builder), so its ifmap count is for the
        # wrong data. The real layer streams C planes, one per column;
        # count that input as read once per element (same element units
        # as SCALE-Sim's counts), 0 when it's already in the SPM. Scaling
        # SCALE-Sim's one-plane count by C instead was tried and measured
        # to over-count 5-22x on MobileNet (the one-plane stream gets
        # re-read once per column fold). Compute, filter and ofmap come
        # from SCALE-Sim as usual -- those are right for this mapping.
        in_shape = tensor_shapes[_activation_input_tensor_id(layer)]['shape']
        ifmap_dram = 0 if ifmap_resident else math.prod(in_shape)
    return {
        'compute_cycles': int(total_cycles),
        # kept split out (rather than pre-summed) so callers can attribute
        # each component separately -- see run_cosma.py's module docstring.
        'ifmap_dram_bytes': ifmap_dram,
        'filter_dram_bytes': int(detail_items[14]),
        'ofmap_dram_bytes': int(detail_items[17]),
    }


def _run_layers(model_json_path: str, config_path: str,
                 topology_csv_path: str, verbose: bool,
                 resident_action: dict = None, spm_plan: dict = None,
                 tensors: dict = None, memory_budget_bytes: int = None,
                 schedule: List[Tuple[int, int]] = None) -> Dict[int, dict]:
    """
    schedule: optional [(t, layer_id), ...] execution order, sorted by t --
        from cosma_Ilp.extract_results()['schedule_layer_at_t'].items()
        under free_schedule=True (see cosma_Ilp.build_cosma_model()). None
        (the default -- always what run_baseline() uses) falls back to
        model.json's own layer order, t == position == layer id, exactly
        today's behavior. This is what lets run_cosma_aware() actually
        simulate a reordered schedule instead of always assuming
        t == layer_id: SpmAllocator.step() must be called with the same
        abstract t that resident_action/spm_plan were keyed by, in
        strictly increasing order, which is no longer model.json's own
        layer order once the schedule is free.
    """
    if resident_action is not None:
        assert spm_plan is not None and tensors is not None and memory_budget_bytes is not None, (
            "run_cosma_aware()'s plan (resident_action/spm_plan/tensors/"
            "memory_budget_bytes) must be supplied together")

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
    # giving less traffic than a bigger one). COSMA's ILP's own decisions are
    # untouched and never see any of this.
    #
    # budget_overflow_events is kept as a frozen historical benchmark: it
    # compares the layer's full filter bytes against what's resident, i.e.
    # "would this layer's whole weight tensor fit next to the residents."
    budget_overflow_events = []
    ifmap_ceiling_events = []
    ofmap_ceiling_events = []
    filter_ceiling_events = []

    row_to_stats: Dict[int, dict] = {}
    for t, lid in schedule:
        layer = layer_by_id[lid]
        if allocator is not None:
            # Replay every timestep's C/P/S/R transitions, including
            # non-conv ones (ADD, DENSE, ...) below -- a tensor can be
            # legitimately resident/spilled through a non-conv timestep
            # too, and skipping those would silently corrupt the
            # allocator's live state. Raises SpmAllocationError loudly if
            # the solved plan is ever physically inconsistent -- see
            # spm_allocator.py's module docstring. Called with t (the
            # abstract schedule timestep resident_action/spm_plan are
            # keyed by), not lid -- those differ once schedule is free.
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

            # An ifmap already in the SPM (resident) needs no
            # working room; everything else shares room_all fairly.
            ifmap_in_spm = ifmap_id is not None and resident_action.get((ifmap_id, t)) == 'P'
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
            verbose, resident_action=resident_action)

    if allocator is not None:
        # Deliberately unconditional, not gated on `verbose` -- that flag
        # also enables SCALE-Sim's own internal per-layer tqdm progress
        # bars (single_layer_sim.run() -> service_memory_requests()),
        # which would flood the terminal with dozens of bars just to
        # surface this one cheap, already-computed summary line. This is
        # the one piece of proof-it-actually-ran output a caller gets by
        # default, without opting into engine-level noise for it.
        print(f"[COSMA SPM] verified {allocator.steps_taken()} timesteps, "
              f"peak occupancy {allocator.peak_occupied_bytes()}/{memory_budget_bytes} "
              f"bytes, 0 violations")
        if budget_overflow_events:
            worst_t, worst_lid, worst_over = max(budget_overflow_events, key=lambda e: e[2])
            print(f"[COSMA SPM] WARNING: {len(budget_overflow_events)} of {len(schedule)} "
                  f"timestep(s) would exceed the {memory_budget_bytes}-byte budget under "
                  f"the OLD, fully-unconstrained filter/weight treatment (layer's full "
                  f"filter bytes on top of what's resident; see the filter line "
                  f"below); worst "
                  f"case {worst_over} bytes over at t={worst_t} (layer {worst_lid})")
        else:
            print(f"[COSMA SPM] budget-vs-working-set check: 0 of {len(schedule)} "
                  f"timesteps exceed the {memory_budget_bytes}-byte budget")
        _print_share_events('COSMA', 'ifmap', ifmap_ceiling_events, len(schedule))
        _print_share_events('COSMA', 'ofmap', ofmap_ceiling_events, len(schedule))
        _print_share_events('COSMA', 'filter', filter_ceiling_events, len(schedule))

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
                     spm_plan: dict, tensors: dict, memory_budget_bytes: int,
                     topology_csv_path: str = None,
                     verbose: bool = True,
                     schedule: List[Tuple[int, int]] = None) -> Dict[int, dict]:
    """
    Same per-layer numbers as run_baseline(), except driven by COSMA's
    actual plan (resident_action, from cosma_Ilp.extract_results()): a
    layer's ifmap fetch is genuinely simulated as free when its
    activation input is resident via 'P', and every layer's ofmap write
    is genuinely simulated as staying on-chip (never drained to DRAM) --
    see _simulate_layer()'s docstring. Unlike run_baseline(), this does
    depend on the SPM budget (indirectly, via which resident_action was
    solved for) and must be re-run per budget.

    schedule: see _run_layers()'s docstring -- required (not None) whenever
    resident_action came from a free_schedule=True solve, since t no longer
    equals layer id in that case; omit it (or pass None) for a
    free_schedule=False plan, where model.json's own order is correct.

    spm_plan/tensors/memory_budget_bytes (also from extract_results(), plus
    graph_builder.load_graph()'s own tensors dict, plus the same budget the
    ILP was solved for) drive a live SpmAllocator replay alongside the real
    SCALE-Sim simulation -- an independent, physically-checked verification
    that resident_action's claims are actually realizable at this budget,
    not just trusted. Required (no default) so this doesn't silently keep
    running without that check after this signature change -- see
    spm_allocator.py's module docstring for why it exists.
    """
    return _run_layers(model_json_path, config_path, topology_csv_path, verbose,
                        resident_action=resident_action, spm_plan=spm_plan,
                        tensors=tensors, memory_budget_bytes=memory_budget_bytes,
                        schedule=schedule)


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
