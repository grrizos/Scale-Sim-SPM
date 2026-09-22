"""Model (topology) registry for the vanilla-vs-optimized benchmark sweep.

Paths are relative to a SCALE-Sim repo root and joined against each
version's --*-repo at call time, since vanilla and optimized are separate
checkouts.
"""

LAYOUT_FILE = "layouts/conv_nets/test.csv"
# Shared for every run: IfmapCustomLayout/FilterCustomLayout are False in
# every combo this sweep uses, so the layout file's content is never read.

MODELS = {
    "googlenet": dict(
        topology="topologies/conv_nets/Googlenet.csv",
        input_type="conv",
    ),
    "resnet18": dict(
        topology="topologies/conv_nets/Resnet18.csv",
        input_type="conv",
    ),
    "resnet50": dict(
        topology="topologies/mlperf/Resnet50.csv",
        input_type="conv",
    ),
    # topologies/conv_nets/Resnet50.csv has extra precomputed Eh/Ew/e2
    # columns and is structurally incompatible with load_arrays_conv's
    # parsing (crashes with IndexError on the first data row -- confirmed
    # empirically). topologies/mlperf/Resnet50.csv has the same clean
    # 8-column header as googlenet/resnet18 and parses fine (54 layers,
    # verified column-count-consistent row by row).
    "vit_b": dict(
        topology="topologies/ispass25_models/vit_b.csv",
        input_type="gemm",  # M/N/K header, confirmed different from the conv models
    ),
    "dense_sparse": dict(
        topology="topologies/sparsity/gemm.csv",
        input_type="gemm",
        sparsity_support=True,
    ),
    # This is SCALE-Sim's own documented sparsity-feature fixture
    # (README_Sparsity.md), not a real network -- 2 tiny GEMM layers with
    # the extra "Sparsity" (N:M) column the sparsity code path expects.
    # Exercises the vanilla-vs-optimized comparison under the sparse
    # metadata-handling logic specifically, distinct from the dense path
    # every other model here takes.
    "vgg16": dict(
        topology="topologies/conv_nets/VGG16.csv",
        input_type="conv",
    ),
    "mobilenetv2": dict(
        topology="topologies/conv_nets/MobileNetV2.csv",
        input_type="conv",
    ),
    "mnasnet": dict(
        topology="topologies/conv_nets/MnasNet.csv",
        input_type="conv",
    ),
    "alexnet": dict(
        topology="topologies/conv_nets/alexnet.csv",
        input_type="conv",
    ),
    "squeezenet": dict(
        topology="topologies/conv_nets/SqueezeNet.csv",
        input_type="conv",
    ),
    # alexnet.csv already existed as a clean, previously-validated native
    # SCALE-Sim topology (no _exported conversion needed -- 5 layers,
    # ran successfully in the original 354-row sweep).
    #
    # SqueezeNet: cosma/_exported/squeezenet/model.json is broken -- every
    # layer (including pooling) is mislabeled 'op': 'ADD' instead of
    # CONV2D/MAXPOOL, so topology_builder.py can't tell conv from pooling
    # from bias-add there (confirmed empirically: it silently wrote 0
    # conv-like rows). The other export,
    # squeezenet_small_cifar100_int8_tucker_svd_5, has correct op labels
    # but is a compressed CIFAR-100 variant, not standard SqueezeNet.
    # Topology instead hand-derived from torchvision's real squeezenet.py
    # source (github.com/pytorch/vision, squeezenet1_0: Fire-module
    # squeeze/expand1x1/expand3x3 channel counts, conv1 k7s2 valid-padded,
    # 3 ceil_mode maxpools), walking PyTorch's actual conv/pool output-size
    # formulas layer by layer (not SAME-padding, unlike the other
    # conversions here -- SqueezeNet's own conv1/Fire convs use explicit
    # padding=0 or 1, not TF-style auto-pad). 26 conv-like layers (conv1
    # stem + 8 Fire modules x 3 convs each + 1x1 classifier conv =
    # 1 + 8*3 + 1 = 26), final spatial size 13x13 before the classifier
    # conv -- both match the known architecture.
    #
    # VGG16/MobileNetV2 topologies were built from cosma/_exported/{VGG16,
    # MobileNetV2}/model.json via cosma/helpers/topology_builder.py (only
    # CONV2D/DEPTHWISE_CONV2D layers get a topology row, SAME-padding
    # folded into IFMAP dims -- see that script's docstring; pulled from
    # the `OnSram` branch since this branch doesn't carry cosma/helpers/
    # itself). VGG16 -> 13 conv layers, matching its known architecture
    # exactly. MobileNetV2 -> 52 conv-like layers.
    #
    # MnasNet had no model.json export available, so its topology was
    # instead hand-derived from torchvision's own mnasnet.py source
    # (alpha=1.0 / mnasnet1_0 config: depths [32,16,24,40,80,96,192,320],
    # stack repeats [3,3,3,2,4,1] -- fetched from
    # github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py,
    # not from memory), walking the same _InvertedResidual/_stack layer
    # unrolling and the same SAME-padding shape math topology_builder.py
    # uses elsewhere. Comes out to 52 conv-like layers and a 7x7 final
    # spatial size (224 / 2^5 stride-2 layers = 7), both consistent with
    # the published architecture -- a real correctness cross-check, not
    # just plausible-looking numbers.
}

MAIN_GRID_MODELS = list(MODELS.keys())

# cProfile subset: 2 models x combos.PROFILE_COMBO_IDS x 2 versions.
PROFILE_MODELS = ["resnet18", "resnet50"]
