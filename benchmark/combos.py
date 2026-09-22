"""Single source of truth for the benchmark parameter grid.

run_sweep.py, run_profile.py, check_correctness.py, and make_plots.py all
import from here so the grid definition never drifts between scripts.
"""

ARRAY_SIZES = [16, 32, 64]
SRAM_KB_VALUES = [16, 32, 64, 128, 256]  # filled in between the original 16/256 endpoints
CALC_GRID_POINTS = [(16, 64), (64, 64)]  # 2 extra CALC corners, matched to USER combos at the same array/sram for a direct comparison

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
    """15 factorial corners (3x5 over array_size x sram_kb, USER bandwidth)
    + 2 matched CALC corners + the anchor (CALC) = 18.

    CALC mostly stays out of the array_size x sram_kb factorial: the
    vanilla-vs-optimized gap (and every observed vanilla timeout) shows up
    under USER, not CALC, so USER gets the full corner sweep while CALC
    gets just the anchor + control combos (sanity-check coverage) plus
    CALC_GRID_POINTS -- 2 corners duplicated at the same array/sram values
    as their USER counterparts, for a direct CALC-vs-USER comparison at
    those specific points."""
    combos = {"anchor": dict(ANCHOR)}
    for array_size in ARRAY_SIZES:
        for sram_kb in SRAM_KB_VALUES:
            combo_id = f"grid_a{array_size}_s{sram_kb}_user"
            combos[combo_id] = dict(
                ANCHOR,
                array_size=array_size,
                sram_kb=sram_kb,
                interface_bandwidth="USER",
            )
    for array_size, sram_kb in CALC_GRID_POINTS:
        combo_id = f"grid_a{array_size}_s{sram_kb}_calc"
        combos[combo_id] = dict(
            ANCHOR,
            array_size=array_size,
            sram_kb=sram_kb,
            interface_bandwidth="CALC",
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
    """All 20 combos run for every model in the main grid (dataflow == 'ws')."""
    combos = main_grid_combos()
    combos.update(control_combos())
    return combos


# Dataflow-generalization addendum: confirms the fix helps in the other 2
# changed compute files (systolic_compute_os.py / _is.py), on 2 cheap
# models only -- not crossed with the full speed grid.
DATAFLOW_ADDENDUM_MODELS = ["resnet18", "vit_b"]
DATAFLOW_ADDENDUM_EXTRA_DATAFLOWS = ["os", "is"]  # 'ws' already covered by the anchor row


def dataflow_addendum_combos():
    """2 combos (one per extra dataflow), anchor settings otherwise."""
    return {
        f"anchor_df{dataflow}": dict(ANCHOR, dataflow=dataflow)
        for dataflow in DATAFLOW_ADDENDUM_EXTRA_DATAFLOWS
    }


# The cProfile subset reuses two combo_ids straight out of the main grid --
# "anchor" and the array64+sram16KB+USER stress corner (biggest measured
# vanilla-vs-optimized gap, see results.csv) -- rather than inventing
# separate definitions that could drift from the main grid.
PROFILE_COMBO_IDS = ["anchor", "grid_a64_s16_user"]
