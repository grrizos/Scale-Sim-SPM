"""Model (topology) registry for the vanilla-vs-optimized benchmark sweep.

Paths are relative to a SCALE-Sim repo root and joined against each
version's --*-repo at call time, since vanilla and optimized are separate
checkouts.
"""

LAYOUT_FILE = "layouts/conv_nets/test.csv"
# Shared for every run: IfmapCustomLayout/FilterCustomLayout are False in
# every combo this sweep uses, so the layout file's content is never read.

MODELS = {
    "alexnet": dict(
        topology="topologies/conv_nets/alexnet.csv",
        input_type="conv",
    ),
    "mobilenet": dict(
        topology="topologies/conv_nets/mobilenet.csv",
        input_type="conv",
    ),
    "resnet50": dict(
        topology="topologies/conv_nets/Resnet50.csv",
        input_type="conv",
    ),
    "vit_b": dict(
        topology="topologies/ispass25_models/vit_b.csv",
        input_type="gemm",  # M/N/K header, confirmed different from the conv models
    ),
    "llama3b": dict(
        topology="topologies/llama/llama3b.csv",
        input_type="conv",  # uses the conv-style IFMAP/Filter header despite being an LLM
    ),
}

MAIN_GRID_MODELS = list(MODELS.keys())

# cProfile subset: 2 models x combos.PROFILE_COMBO_IDS x 2 versions.
PROFILE_MODELS = ["alexnet", "llama3b"]
