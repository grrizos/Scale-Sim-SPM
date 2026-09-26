# cosma/helpers/topology_builder.py
"""
Builds a SCALE-Sim topology CSV from model.json's CONV2D/DEPTHWISE_CONV2D
layers. SCALE-Sim only simulates conv-like (GEMM-mappable) layers -- ADD,
DENSE (handled separately by SCALE-Sim's own GEMM path), REDUCE_MEAN, etc.
have no topology row and get 0 compute/DRAM cost from SCALE-Sim's side.

Depthwise conv has no native representation in SCALE-Sim's topology format
(no "groups" concept). A real accelerator maps it channels-across-columns:
each array column holds one channel's kh x kw filter and works on that
channel's input plane. The timing of that mapping is exactly what SCALE-Sim
computes for a row with Channels = 1 and Num Filter = C (kh*kw array rows,
C columns in ceil(C / ArrayWidth) folds), so that's the row written here.
SCALE-Sim then gets compute cycles, filter traffic (kh*kw*C) and ofmap
traffic (H'*W'*C) right. The one thing that row gets wrong is the input:
SCALE-Sim sends one input plane to every column, while real depthwise reads
C different planes -- the runner's _simulate_layer() replaces the ifmap
DRAM count for depthwise layers with the real input tensor, each element
read once.
(Previously written as Channels = C, Num Filter = 1, the
topologies/conv_nets/mobilenet.csv convention: right MAC count, but it
simulates a filter summing all C channels into one output channel -- 1 of
the array's columns busy, 1-channel ofmap traffic.)
"""
import csv
import json
import math
from typing import Dict, List, Tuple


def _same_padded_dim(in_dim: int, k: int, stride: int) -> int:
    """
    SCALE-Sim's topology CSV has no padding field -- it expects IFMAP
    dimensions with any 'SAME' padding already folded in (this is why
    TFLite exporters often insert an explicit PAD layer). Reconstruct the
    padded size using the standard TF SAME-padding formula.
    """
    out_dim = math.ceil(in_dim / stride)
    pad_total = max((out_dim - 1) * stride + k - in_dim, 0)
    return in_dim + pad_total


def build_topology(model_json_path: str, csv_path: str) -> Dict[int, int]:
    """
    Writes a SCALE-Sim topology CSV to csv_path.
    Returns layer_id_to_row: model.json layer id -> topology CSV row index
    (0-based, matching SCALE-Sim's per-layer report ordering).
    """
    with open(model_json_path, 'r') as f:
        model = json.load(f)

    rows: List[Tuple] = []
    layer_id_to_row: Dict[int, int] = {}

    for layer in model['layers']:
        op = layer['op']
        if op not in ('CONV2D', 'DEPTHWISE_CONV2D'):
            continue

        params = layer['params']
        in_shape = layer['input_shape']    # [N, H, W, C]
        out_shape = layer['output_shape']  # [N, H, W, C]
        ifmap_h, ifmap_w = in_shape[1], in_shape[2]
        kh, kw = params['kh'], params['kw']
        stride = params.get('stride_h', 1)

        if params.get('pad') == 'SAME':
            ifmap_h = _same_padded_dim(ifmap_h, kh, stride)
            ifmap_w = _same_padded_dim(ifmap_w, kw, stride)

        if op == 'DEPTHWISE_CONV2D':
            # Channels-across-columns (see module docstring). Assumes
            # channel multiplier 1 (output channels == input channels),
            # true for every depthwise layer in this repo's models.
            assert out_shape[3] == in_shape[3], (
                f"layer {layer['id']}: depthwise channel multiplier != 1 not supported")
            channels = 1
            num_filters = in_shape[3]
        else:
            channels = in_shape[3]
            num_filters = out_shape[3]

        row_index = len(rows)
        layer_id_to_row[layer['id']] = row_index
        name = f"{op}_{layer['id']}"
        rows.append((name, ifmap_h, ifmap_w, kh, kw, channels, num_filters, stride))

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer name', ' IFMAP Height', ' IFMAP Width',
                          ' Filter Height', ' Filter Width', ' Channels',
                          ' Num Filter', ' Strides', ''])
        for r in rows:
            writer.writerow(list(r) + [''])

    return layer_id_to_row


if __name__ == '__main__':
    import argparse
    import os

    # cosma/ (one level up from helpers/), where model.json/topology.csv
    # live by default.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-json', default=os.path.join(here, 'model.json'),
                         help='Path to model.json (default: cosma/model.json).')
    parser.add_argument('--out-csv', default=os.path.join(here, 'topology.csv'),
                         help='Path to write the topology CSV to '
                              '(default: cosma/topology.csv).')
    args = parser.parse_args()

    mapping = build_topology(args.model_json, args.out_csv)
    print(f"Wrote {len(mapping)} conv-like layers to {args.out_csv}")
    print(f"First few mappings (layer id -> row): "
          f"{dict(list(mapping.items())[:5])}")
