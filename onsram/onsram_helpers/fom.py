# onsram/onsram_helpers/fom.py
"""
Figure-of-Merit scoring for OnSRAM-Static, ported from
/home/george/trim/spm_management/onsram_weights.py (functions
_estimate_flops, compute_reuse_factor, _infer_node_type,
calculate_fom_corrected -- the OnSRAM paper-faithful FoM variant,
confirmed against the real OnSRAM paper's Eq. 1 in
onsram/docs/onsram_integration_plan.md section 4, not the ad-hoc
`calculate_fom` the reference's own active pipeline actually calls).

spm_common/graph_builder.py's Node dataclass (id, op, activation_inputs,
weight_inputs, outputs) deliberately does not carry input_shape/
output_shape/params -- fields the reference's FLOPs/reuse/node-type logic
all need. Rather than touching graph_builder.py, this
module separately re-reads the same model.json (load_layer_meta below)
for just those fields. All OnSRAM-specific derived state (flops, reuse
metadata, node type, fom) lives in plain dicts keyed by id here -- COSMA's
Node/Tensor dataclasses are never mutated.

Preserved reference quirk: `_infer_node_type`'s arithmetic-intensity
byte count is hardcoded to 2 bytes/element ("force float16") regardless
of a tensor's real dtype (onsram_weights.py line 670, and its call site
already always passes the literal "float16" regardless of dtype -- line
419). This changes node-type classification results for fp32 models but
NOT tensor byte sizes (those come from graph_builder.py's own real-dtype
compute_size_bytes). Reproduced verbatim here, not "fixed", since this
port is validated against the reference's own classification decisions.
"""
import json
import math
from enum import Enum
from typing import Dict, List, Tuple


class NodeType(Enum):
    ACTIVATION_BOUND = "activation"
    COMPUTE_BOUND = "compute"
    WEIGHT_BOUND = "weight"


def load_layer_meta(model_json_path: str) -> Dict[int, dict]:
    """
    Bare json.load side-read of model.json's raw layer records, keyed by
    layer id. Does NOT duplicate any of graph_builder.py's tensor/
    producer/consumer parsing -- only reads the handful of fields
    graph_builder.Node drops (input_shape/output_shape/params/op).
    """
    with open(model_json_path, 'r') as f:
        model = json.load(f)
    return {
        layer['id']: {
            'op': layer['op'],
            'input_shape': layer.get('input_shape', [1, 1, 1, 1]),
            'output_shape': layer.get('output_shape', [1, 1, 1, 1]),
            'params': layer.get('params', {}),
        }
        for layer in model['layers']
    }


def _shape_prod(shape) -> int:
    return math.prod(shape) if shape else 1


def estimate_node_flops(layer_meta: Dict[int, dict]) -> Dict[int, float]:
    """
    Direct port of _estimate_flops(): per-op-type MACs table, x2 for
    FLOPs. {node_id: flops}.
    """
    node_flops: Dict[int, float] = {}
    for node_id, meta in layer_meta.items():
        op_upper = meta['op'].upper()
        input_shape = meta['input_shape']
        output_shape = meta['output_shape']
        params = meta['params']

        out_elements = _shape_prod(output_shape)
        input_channels = input_shape[-1] if input_shape else 1

        if op_upper == "CONV2D":
            kh = params.get("kh", 3)
            kw = params.get("kw", 3)
            macs = float(out_elements * kh * kw * input_channels)
        elif op_upper == "DEPTHWISE_CONV2D":
            kh = params.get("kh", 3)
            kw = params.get("kw", 3)
            macs = float(out_elements * kh * kw)
        elif op_upper in ("DENSE", "FC", "FULLY_CONNECTED"):
            macs = float(out_elements * input_channels)
        elif op_upper in ("MAXPOOL", "AVGPOOL"):
            kh = params.get("kh", 2)
            kw = params.get("kw", 2)
            macs = float(out_elements * kh * kw)
        elif op_upper in ("ADD", "SUB", "MUL", "RELU", "RELU6", "SIGMOID", "TANH"):
            macs = float(out_elements * 0.5)
        elif op_upper in ("CONCAT", "RESHAPE", "PAD"):
            macs = 0.0
        else:
            macs = float(out_elements)

        node_flops[node_id] = macs * 2.0
    return node_flops


def build_reuse_meta(layer_meta: Dict[int, dict]) -> Dict[int, dict]:
    """
    Direct port of load_graph_from_json()'s per-node metadata stamping
    (onsram_weights.py lines 374-407): the {kh, kw, cout, cin, op} bundle
    compute_reuse_factor() reads via getattr(node, ...). Same defaults as
    the reference: output_channels=1, kh=1, kw=1, cin=1 for any op not
    explicitly handled below (note: NOT kh=kw=3 -- that default only
    applies inside _estimate_flops's own params.get() calls, a different
    function with its own separate defaults).
    """
    reuse_meta: Dict[int, dict] = {}
    for node_id, meta in layer_meta.items():
        op_upper = meta['op'].upper()
        input_shape = meta['input_shape']
        output_shape = meta['output_shape']
        params = meta['params']

        cout, kh, kw, cin = 1, 1, 1, 1

        if op_upper == "CONV2D":
            cout = output_shape[-1] if output_shape else 1
            kh = params.get('kh', 3)
            kw = params.get('kw', 3)
            cin = input_shape[-1] if input_shape else 1
        elif op_upper == "DEPTHWISE_CONV2D":
            kh = params.get('kh', 3)
            kw = params.get('kw', 3)
            cin = input_shape[-1] if input_shape else 1
        elif op_upper in ("MAXPOOL", "AVGPOOL", "GLOBALAVGPOOL"):
            kh = params.get('kh', params.get('kernel_size', 2))
            kw = params.get('kw', params.get('kernel_size', 2))
        elif op_upper in ("DENSE", "FC", "FULLY_CONNECTED"):
            cout = output_shape[-1] if output_shape else 1
            cin = input_shape[-1] if input_shape else 1

        reuse_meta[node_id] = {
            'op': op_upper, 'kh': int(kh), 'kw': int(kw),
            'cout': int(cout), 'cin': int(cin),
        }
    return reuse_meta


def compute_reuse_factor(consumer_node_id: int, reuse_meta: Dict[int, dict]) -> float:
    """
    Direct id-keyed port of compute_reuse_factor(): structural per-op
    reuse (Conv: Kh*Kw*Cout, Depthwise: Kh*Kw, Dense: Cin, everything
    else: 1). The reference's own signature takes an unused `tensor`
    argument (reuse depends only on the consuming node, never the tensor
    itself) -- dropped here since nothing reads it.
    """
    if consumer_node_id not in reuse_meta:
        return 1.0
    meta = reuse_meta[consumer_node_id]
    op = meta['op']

    if op == "CONV2D":
        return float(meta['kh'] * meta['kw'] * meta['cout'])
    elif op == "DEPTHWISE_CONV2D":
        return float(meta['kh'] * meta['kw'])
    elif op in ("RELU", "RELU6", "SIGMOID", "TANH", "ELU",
                "MAXPOOL", "AVGPOOL", "GLOBALAVGPOOL"):
        return 1.0
    elif op in ("ADD", "SUB", "MUL", "DIV", "MOD"):
        return 1.0
    elif op in ("CONCAT", "RESHAPE", "PAD", "SLICE", "TRANSPOSE"):
        return 1.0
    elif op in ("DENSE", "FC", "FULLY_CONNECTED"):
        return float(meta['cin'])
    else:
        return 1.0


def infer_node_type(node_id: int, layer_meta: Dict[int, dict],
                     node_flops: Dict[int, float]) -> NodeType:
    """
    Direct port of _infer_node_type()'s roofline classification
    (arithmetic intensity vs. thresholds 93.75 for dense-array ops /
    11.72 for SFU ops; Dense/FC/MatMul always WEIGHT_BOUND). Preserves
    the hardcoded bytes_per_element = 2 quirk -- see module docstring.
    """
    meta = layer_meta[node_id]
    op_upper = meta['op'].upper()

    if op_upper in ("DENSE", "FC", "FULLY_CONNECTED", "MATMUL"):
        return NodeType.WEIGHT_BOUND

    input_shape = meta['input_shape']
    output_shape = meta['output_shape']
    params = meta['params']
    flops = node_flops[node_id]

    bytes_per_element = 2  # Force float16 for OnSRAM -- see module docstring.

    input_elements = _shape_prod(input_shape)
    output_elements = _shape_prod(output_shape)
    activation_bytes = (input_elements + output_elements) * bytes_per_element

    weight_bytes = 0.0
    if op_upper == "CONV2D":
        cin = input_shape[-1] if input_shape else 1
        cout = output_shape[-1] if output_shape else 1
        kh = params.get('kh', 3)
        kw = params.get('kw', 3)
        weight_bytes = float(cin * cout * kh * kw * bytes_per_element)
    elif op_upper == "DEPTHWISE_CONV2D":
        cin = input_shape[-1] if input_shape else 1
        kh = params.get('kh', 3)
        kw = params.get('kw', 3)
        weight_bytes = float(cin * kh * kw * bytes_per_element)

    total_bytes_moved = activation_bytes + weight_bytes
    arithmetic_intensity = flops / total_bytes_moved if total_bytes_moved > 0 else 0.0

    if op_upper in ("CONV2D", "DEPTHWISE_CONV2D", "DENSE", "FC", "MATMUL"):
        threshold = 93.75
    else:
        threshold = 11.72

    if arithmetic_intensity > threshold:
        return NodeType.COMPUTE_BOUND
    else:
        return NodeType.ACTIVATION_BOUND


def classify_all_nodes(nodes: Dict[int, object], layer_meta: Dict[int, dict],
                        node_flops: Dict[int, float]) -> Dict[int, NodeType]:
    return {node_id: infer_node_type(node_id, layer_meta, node_flops)
            for node_id in nodes}


def calculate_fom(nodes: Dict[int, object], tensors: Dict[int, object],
                   schedule: List[Tuple[int, int]],
                   reuse_meta: Dict[int, dict],
                   node_flops: Dict[int, float],
                   produced_at_ts: Dict[int, int],
                   last_used_at_ts: Dict[int, int]) -> Dict[int, float]:
    """
    Direct id-keyed port of calculate_fom_corrected() (paper Eq. 1,
    alpha=0.4/beta=0.5/gamma=0.1). `schedule` is COSMA's
    [(t, layer_id), ...] convention. Returns {tensor_id: fom} -- a plain
    side-dict, never stored on Tensor.
    """
    alpha, beta, gamma = 0.4, 0.5, 0.1
    n_nodes = len(nodes)
    ts_map = {layer_id: t for t, layer_id in schedule}
    total_ops = sum(node_flops.values())

    tensor_fom: Dict[int, float] = {}
    for tensor_id, tensor in tensors.items():
        produced = produced_at_ts[tensor_id]
        last_used = last_used_at_ts[tensor_id]
        liveness_span = last_used - produced

        timesteps_used = {ts_map[c] for c in tensor.consumer_layers if c in ts_map}
        unused_liveness = max(0, liveness_span - len(timesteps_used))
        ul_term = alpha * (n_nodes / (unused_liveness + 1.0))

        mb_sum = 0.0
        for consumer_id in tensor.consumer_layers:
            if consumer_id not in nodes:
                continue
            reuse = compute_reuse_factor(consumer_id, reuse_meta)
            if reuse > 0:
                mb_sum += 1.0 / reuse
        mb_term = beta * mb_sum

        consumer_ops = sum(node_flops[c] for c in tensor.consumer_layers if c in nodes)
        impact_term = gamma * (consumer_ops / total_ops if total_ops > 0 else 0.0)

        tensor_fom[tensor_id] = ul_term + mb_term + impact_term

    return tensor_fom
