"""Single source of truth for the benchmark parameter grid.

run_sweep.py, run_profile.py, check_correctness.py, and make_plots.py all
import from here so the grid definition never drifts between scripts.
"""

ARRAY_SIZES = [16, 64]
SRAM_KB_VALUES = [16, 256]
BANDWIDTH_MODES = ["CALC", "USER"]

ANCHOR = dict(
    array_size=32,
    sram_kb=64,
    interface_bandwidth="CALC",
    dataflow="ws",
    ifmap_offset=0,
    filter_offset=10_000_000,
    ofmap_offset=20_000_000,
    rng_seed=40,
)


def main_grid_combos():
    """8 factorial corners (2x2x2 over the 3 speed params) + the anchor = 9."""
    combos = {"anchor": dict(ANCHOR)}
    for array_size in ARRAY_SIZES:
        for sram_kb in SRAM_KB_VALUES:
            for bw in BANDWIDTH_MODES:
                combo_id = f"grid_a{array_size}_s{sram_kb}_{bw.lower()}"
                combos[combo_id] = dict(
                    ANCHOR,
                    array_size=array_size,
                    sram_kb=sram_kb,
                    interface_bandwidth=bw,
                )
    return combos


def control_combos():
    """2 non-speed control combos, each a single delta against the anchor
    (not crossed with the speed grid -- a deliberate scope cut)."""
    return {
        "ctrl_offset_shift": dict(
            ANCHOR,
            ifmap_offset=5_000_000,
            filter_offset=15_000_000,
            ofmap_offset=25_000_000,
        ),
        "ctrl_seed_alt": dict(ANCHOR, rng_seed=777),
    }


def all_main_combos():
    """All 11 combos run for every model in the main grid (dataflow == 'ws')."""
    combos = main_grid_combos()
    combos.update(control_combos())
    return combos


# Dataflow-generalization addendum: confirms the fix helps in the other 2
# changed compute files (systolic_compute_os.py / _is.py), on 2 cheap
# models only -- not crossed with the full speed grid.
DATAFLOW_ADDENDUM_MODELS = ["alexnet", "mobilenet"]
DATAFLOW_ADDENDUM_EXTRA_DATAFLOWS = ["os", "is"]  # 'ws' already covered by the anchor row


def dataflow_addendum_combos():
    """2 combos (one per extra dataflow), anchor settings otherwise."""
    return {
        f"anchor_df{dataflow}": dict(ANCHOR, dataflow=dataflow)
        for dataflow in DATAFLOW_ADDENDUM_EXTRA_DATAFLOWS
    }


# The cProfile subset reuses two combo_ids straight out of the main grid --
# "anchor" and the array64+sram16KB+CALC stress corner -- rather than
# inventing separate definitions that could drift from the main grid.
PROFILE_COMBO_IDS = ["anchor", "grid_a64_s16_calc"]
