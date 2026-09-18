"""Writes a SCALE-Sim .cfg file for a given parameter combo.

Every field not explicitly parameterized here is pinned at the values
shipped in configs/scale.cfg, so vanilla and optimized always see an
identical config for a given combo -- the only difference is which
repo's scalesim/scale.py consumes it. Never reads either repo's own
checked-in configs/scale.cfg; everything comes from this template.
"""
import os

_TEMPLATE = """[general]
run_name = {run_id}

[architecture_presets]
ArrayHeight:    {array_size}
ArrayWidth:     {array_size}
IfmapSramSzkB:   {sram_kb}
FilterSramSzkB:  {sram_kb}
OfmapSramSzkB:   {sram_kb}
IfmapOffset:    {ifmap_offset}
FilterOffset:   {filter_offset}
OfmapOffset:    {ofmap_offset}
Bandwidth : 10
Dataflow : {dataflow}
MemoryBanks:   1
ReadRequestBuffer: 32
WriteRequestBuffer: 32

[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: 10
IfmapSRAMBankNum: 10
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: 10
FilterSRAMBankNum: 10
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport : false
SparseRep : ellpack_block
OptimizedMapping : false
BlockSize : 8
RandomNumberGeneratorSeed : {rng_seed}

[run_presets]
InterfaceBandwidth: {interface_bandwidth}
UseRamulatorTrace: False
"""


def write_config(cfg_dir, run_id, *, array_size, sram_kb, interface_bandwidth,
                  dataflow, ifmap_offset, filter_offset, ofmap_offset, rng_seed):
    """Writes <cfg_dir>/<run_id>.cfg and returns its path."""
    os.makedirs(cfg_dir, exist_ok=True)
    path = os.path.join(cfg_dir, run_id + ".cfg")
    with open(path, "w") as f:
        f.write(_TEMPLATE.format(
            run_id=run_id,
            array_size=array_size,
            sram_kb=sram_kb,
            ifmap_offset=ifmap_offset,
            filter_offset=filter_offset,
            ofmap_offset=ofmap_offset,
            dataflow=dataflow,
            rng_seed=rng_seed,
            interface_bandwidth=interface_bandwidth,
        ))
    return path
