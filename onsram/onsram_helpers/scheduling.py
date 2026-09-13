# onsram/onsram_helpers/scheduling.py
"""
BFS-DFS hybrid scheduling + liveness analysis, ported from
/home/george/trim/spm_management/onsram_weights.py's
schedule_nodes_bfs_dfs_hybrid() and analyze_liveness().

Id-keyed translation of the reference's string-name-keyed graph, using
COSMA's own field names (cosma/helpers/graph_builder.py): a Node's
`activation_inputs` is already exactly "tensor ids produced by another
layer" by construction, so (unlike the reference, which double-checks
`producer in graph.nodes` while walking every input including weights)
this port's in-degree/children computation can skip that filtering --
weight_inputs are structurally never activation dependencies here.
"""
from collections import deque
from typing import Dict, List, Tuple

from .fom import NodeType


def compute_in_degree(nodes: Dict[int, object], tensors: Dict[int, object]) -> Dict[int, int]:
    """
    Direct id-keyed port of schedule_nodes_bfs_dfs_hybrid()'s Step 1.
    in_degree[node_id] = number of distinct producer layers among its
    activation_inputs.
    """
    in_degree: Dict[int, int] = {}
    for node_id, node in nodes.items():
        producers = {tensors[tid].producer_layer for tid in node.activation_inputs
                     if tid in tensors}
        in_degree[node_id] = len(producers)
    return in_degree


def build_children_by_producer(nodes: Dict[int, object],
                                tensors: Dict[int, object]) -> Dict[int, set]:
    """Direct id-keyed port of schedule_nodes_bfs_dfs_hybrid()'s Step 2."""
    children_by_producer: Dict[int, set] = {node_id: set() for node_id in nodes}
    for tensor in tensors.values():
        producer = tensor.producer_layer
        if producer not in children_by_producer:
            continue
        for consumer in tensor.consumer_layers:
            if consumer in nodes and consumer != producer:
                children_by_producer[producer].add(consumer)
    return children_by_producer


def schedule_bfs_dfs_hybrid(nodes: Dict[int, object], tensors: Dict[int, object],
                             node_type: Dict[int, NodeType]) -> List[int]:
    """
    Direct id-keyed port of schedule_nodes_bfs_dfs_hybrid()'s deque-based
    Kahn's-algorithm walk. Newly-ready ACTIVATION_BOUND nodes are pushed
    to the ready-queue HEAD (DFS-style, minimizing producer-to-consumer
    distance); COMPUTE_BOUND/WEIGHT_BOUND nodes to the TAIL (BFS-style).

    The tie-break key reconstructs the reference's exact string sort key
    f"{op}_{node_id}" (e.g. "CONV2D_10" < "DEPTHWISE_CONV2D_9"
    lexicographically, despite 10 > 9) rather than a plain numeric id --
    confirmed by reading the reference this is a real, different
    ordering rule on any graph where two same-priority nodes become
    ready simultaneously (invisible on a linear chain like MobileNet,
    real on a branching graph).
    """
    in_degree = compute_in_degree(nodes, tensors)
    children_by_producer = build_children_by_producer(nodes, tensors)

    def ready_priority(node_id: int) -> Tuple[int, str]:
        nt = node_type[node_id]
        if nt == NodeType.ACTIVATION_BOUND:
            priority = 0
        elif nt == NodeType.COMPUTE_BOUND:
            priority = 1
        else:
            priority = 2
        return priority, f"{nodes[node_id].op}_{node_id}"

    ready = deque(sorted(
        (nid for nid, deg in in_degree.items() if deg == 0),
        key=ready_priority,
    ))
    scheduled = set()
    schedule: List[int] = []

    while ready:
        current = ready.popleft()
        if current in scheduled:
            continue
        scheduled.add(current)
        schedule.append(current)

        newly_ready = []
        for consumer in children_by_producer[current]:
            if consumer in scheduled:
                continue
            in_degree[consumer] -= 1
            if in_degree[consumer] == 0:
                newly_ready.append(consumer)

        newly_ready = sorted(set(newly_ready), key=ready_priority)

        for consumer in reversed([n for n in newly_ready
                                   if node_type[n] == NodeType.ACTIVATION_BOUND]):
            ready.appendleft(consumer)

        for consumer in [n for n in newly_ready
                         if node_type[n] != NodeType.ACTIVATION_BOUND]:
            ready.append(consumer)

    if len(schedule) != len(nodes):
        missing = sorted(set(nodes) - set(schedule))
        raise ValueError(f"Unable to schedule all nodes; graph may contain a cycle: {missing[:5]}")

    return schedule


def analyze_liveness(nodes: Dict[int, object], tensors: Dict[int, object],
                      schedule_order: List[int]
                      ) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Direct id-keyed port of analyze_liveness(). Returns three side-dicts
    (does not mutate tensors, unlike the reference):
      produced_at_ts:  {tensor_id: schedule index of producer_layer, or -1}
      last_used_at_ts: {tensor_id: max schedule index among consumer_layers,
                                    or produced_at_ts if never consumed}
      reuse_count:     {tensor_id: len(consumer_layers)}
    """
    ts_map = {node_id: t for t, node_id in enumerate(schedule_order)}

    produced_at_ts: Dict[int, int] = {}
    last_used_at_ts: Dict[int, int] = {}
    reuse_count: Dict[int, int] = {}

    for tensor_id, tensor in tensors.items():
        produced = ts_map.get(tensor.producer_layer, -1)
        produced_at_ts[tensor_id] = produced

        consumer_timesteps = [ts_map[c] for c in tensor.consumer_layers if c in ts_map]
        last_used_at_ts[tensor_id] = max(consumer_timesteps) if consumer_timesteps else produced

        reuse_count[tensor_id] = len(tensor.consumer_layers)

    return produced_at_ts, last_used_at_ts, reuse_count
