# cosma/helpers/graph_builder.py
"""
Parses model.json (COSMA's tensor-graph input format) into two dicts:
  nodes:   layer id -> Node    (operators)
  tensors: tensor id -> Tensor (activations only; weights/bias/network-input
                                 tensors are excluded -- they are always
                                 compulsory DRAM traffic and are not tracked
                                 by COSMA's SPM ILP)

A tensor is "COSMA-tracked" iff it is produced by some layer's `outputs`.
Any input whose `inputs_from` entry is -1 (weights, bias, or the network's
own raw input) is never produced by a layer, so it is naturally excluded --
we don't need to special-case tensor ids or dtypes to find them.
"""
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Node:
    id: int
    op: str
    activation_inputs: List[int]   # tensor ids produced by another layer
    weight_inputs: List[int]       # tensor ids with inputs_from == -1 (weights/bias/raw input)
    outputs: List[int]


@dataclass
class Tensor:
    id: int
    size_bytes: int
    producer_layer: int
    consumer_layers: List[int] = field(default_factory=list)
    producer_timestep: int = -1     # filled in during scheduling
    last_used_timestep: int = -1    # filled in during scheduling


_DTYPE_BYTES = {
    'float32': 4,
    'float16': 2,
    'bfloat16': 2,
    'int32': 4,
    'int8': 1,
    'uint8': 1,
}


def compute_size_bytes(shape, dtype: str) -> int:
    bytes_per_elem = _DTYPE_BYTES.get(dtype.lower(), 4)
    num_elements = math.prod(shape) if shape else 1
    return num_elements * bytes_per_elem


def load_graph(model_json_path: str):
    with open(model_json_path, 'r') as f:
        model = json.load(f)

    tensor_shapes: Dict[int, dict] = {t['id']: t for t in model['tensors']}

    # Pre-pass: which layer produces each tensor id (O(N), avoids nested loops).
    tensor_producer: Dict[int, int] = {}
    for layer in model['layers']:
        for out_id in layer.get('outputs', []):
            tensor_producer[out_id] = layer['id']

    nodes: Dict[int, Node] = {}
    tensors: Dict[int, Tensor] = {}

    for layer in model['layers']:
        layer_id = layer['id']
        inputs = layer.get('inputs', [])
        inputs_from = layer.get('inputs_from', [-1] * len(inputs))
        outputs = layer.get('outputs', [])

        activation_inputs = [
            tid for tid, src in zip(inputs, inputs_from) if src != -1
        ]
        weight_inputs = [
            tid for tid, src in zip(inputs, inputs_from) if src == -1
        ]

        nodes[layer_id] = Node(
            id=layer_id,
            op=layer['op'],
            activation_inputs=activation_inputs,
            weight_inputs=weight_inputs,
            outputs=outputs,
        )

        # Only tensors this layer *produces* are COSMA-tracked activations.
        for out_id in outputs:
            info = tensor_shapes[out_id]
            tensors[out_id] = Tensor(
                id=out_id,
                size_bytes=compute_size_bytes(info['shape'], info['dtype']),
                producer_layer=layer_id,
            )

    # Second pass: fill in consumer_layers now that all tensors are known.
    for layer in model['layers']:
        for tid in layer.get('inputs', []):
            if tid in tensors:
                tensors[tid].consumer_layers.append(layer['id'])

    return nodes, tensors


def print_graph_summary(nodes: Dict[int, Node], tensors: Dict[int, Tensor]) -> None:
    print(f"Nodes: {len(nodes)}, COSMA-tracked tensors: {len(tensors)}")
    skip = {tid: t for tid, t in tensors.items() if len(t.consumer_layers) > 1}
    print(f"Skip-connection tensors (consumed by >1 layer): {len(skip)}")
    for tid, t in list(skip.items())[:5]:
        print(f"  tensor {tid}: {t.size_bytes} bytes, "
              f"producer=layer {t.producer_layer}, consumers={t.consumer_layers}")


if __name__ == '__main__':
    import os
    # cosma/ (one level up from helpers/), where model.json actually lives.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nodes, tensors = load_graph(os.path.join(here, 'model.json'))
    print_graph_summary(nodes, tensors)
