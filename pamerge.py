from typing import Dict, List, Set, Tuple
import random

from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode
import retworkx as rx

from helper import get_qreg_index
from pulseDB import PulseDB
from qocdag import QOCDAG


def get_merge_candidates(
    qoc_dag: QOCDAG, prune_method: str, maxN: int, pattern_nodes: Set[DAGNode] = []
) -> List[List[DAGNode]]:
    candidate_dict: Dict[Tuple[int], List[DAGNode]] = dict()
    if prune_method == "random":
        dag_circuit = qoc_dag.get_dag_circuit()
        node: DAGNode
        for node in dag_circuit.topological_op_nodes():
            if node in pattern_nodes:
                continue
            succ: DAGNode
            for succ in dag_circuit.successors(node):
                if (
                    len(get_qreg_index([node, succ])) > maxN
                    or not isinstance(succ, DAGOpNode)
                    or succ in pattern_nodes
                    or qoc_dag.check_can_merge([node, succ]) is False
                ):
                    continue
                candidate_dict[tuple(sorted([node._node_id, succ._node_id]))] = [
                    node,
                    succ,
                ]
    elif prune_method == "extended":
        dag_circuit = qoc_dag.get_dag_circuit()
        node: DAGNode
        for node in dag_circuit.topological_op_nodes():
            if node in pattern_nodes:
                continue
            succ: DAGNode
            for succ in dag_circuit.successors(node):
                if (
                    len(get_qreg_index([node, succ])) > maxN
                    or not isinstance(succ, DAGOpNode)
                    or succ in pattern_nodes
                    or qoc_dag.check_can_merge([node, succ]) is False
                ):
                    continue
                extended_nodes = qoc_dag.generate_extended_nodes(
                    [node, succ], pattern_nodes
                )
                # print(set(node.name for node in pattern_nodes))
                # print(node1.name, node2.name)
                if extended_nodes is not None:
                    extended_nodes: List[DAGNode]
                    # print([node.name for node in extended_nodes])
                    candidate_dict[
                        tuple(sorted([node._node_id for node in extended_nodes]))
                    ] = extended_nodes
    elif prune_method == "critical":
        c_path, _ = qoc_dag.get_critical_path()
        for node1, node2 in zip(c_path[:-1], c_path[1:]):
            if (
                len(get_qreg_index([node1, node2])) > maxN
                or not isinstance(node1, DAGOpNode)
                or not isinstance(node2, DAGOpNode)
                or node1 in pattern_nodes
                or node2 in pattern_nodes
            ):
                continue
            extended_nodes = qoc_dag.generate_extended_nodes(
                [node1, node2], pattern_nodes
            )
            # print(set(node.name for node in pattern_nodes))
            # print(node1.name, node2.name)
            if extended_nodes is not None:
                extended_nodes: List[DAGNode]
                # print([node.name for node in extended_nodes])
                candidate_dict[
                    tuple(sorted([node._node_id for node in extended_nodes]))
                ] = extended_nodes
    else:
        raise Exception(f"prune method {prune_method} has not been implemented.")
    # print('-------------')
    # for candidate_id, nodes in candidate_dict.items():
    #     print(candidate_id)
    #     for node in nodes:
    #         print(node.name, end = ' ')
    #     print()
    return list(candidate_dict.values())


def cal_merge_score(qoc_dag: QOCDAG, nodes: List[DAGNode], merge_heuristic: str) -> int:
    if merge_heuristic == "random":
        random.seed(0)
        score = random.randrange(0, 100)
    elif merge_heuristic == "approx":
        # Get base which is 10^N
        base = pow(10, len(get_qreg_index(nodes)))
        c_path, _ = qoc_dag.get_critical_path()
        # Get last node in nodes that is on the critical path
        last_c_node_index = max(
            [i for i, node in enumerate(c_path) if node in nodes], default=-1
        )
        c_node_counts = sum(1 for node in c_path if node in nodes)
        if last_c_node_index == -1:
            # no nodes on the critical path
            return base
        elif last_c_node_index == len(c_path) - 1:
            # this includes the last node on the critical path. return base
            return base
        elif c_node_counts == 1:
            # this means only one gate on the critical path, return base / 2
            return base / 2
        B_prime = c_path[last_c_node_index + 1]
        B_prime_latency, _ = qoc_dag.pulse_db.get_optimal_pulse_time([B_prime])
        all_next_nc_nodes_c: List[DAGNode] = []
        all_next_nc_nodes_nc: List[DAGNode] = []
        for node in nodes:
            for succ in qoc_dag.dag.successors(node):
                if not isinstance(succ, DAGOpNode) or succ in nodes or succ in c_path:
                    continue
                descs = [
                    node
                    for node in qoc_dag.dag.descendants(node)
                    if isinstance(node, DAGOpNode)
                ]
                if B_prime in descs:
                    all_next_nc_nodes_c.append(succ)
                else:
                    all_next_nc_nodes_nc.append(succ)

        if len(all_next_nc_nodes_c) != 0:
            all_next_nc_nodes_c.sort(key=lambda x: qoc_dag.get_node_end_time(x))
            C = all_next_nc_nodes_c[0]
            C_latency, _ = qoc_dag.pulse_db.get_optimal_pulse_time([C])
            return base + B_prime_latency - C_latency
        else:
            if len(all_next_nc_nodes_nc) == 0:
                return base / 2
            all_next_nc_nodes_nc.sort(key=lambda x: qoc_dag.get_node_end_time(x))
            C = all_next_nc_nodes_nc[0]

            B_prime_all_descendants = list(
                set(
                    node
                    for node in qoc_dag.dag.descendants(B_prime)
                    if isinstance(node, DAGOpNode)
                )
            )
            B_prime_all_descendants_end_time = [
                qoc_dag.get_node_end_time(node) for node in B_prime_all_descendants
            ]
            B_prime_l = (
                max(B_prime_all_descendants_end_time)
                - qoc_dag.get_node_end_time(B_prime)
                if len(B_prime_all_descendants_end_time) != 0
                else base / 2
            )

            C_all_descendants = list(
                set(
                    node
                    for node in qoc_dag.dag.descendants(C)
                    if isinstance(node, DAGOpNode)
                )
            )
            C_all_descendants_end_time = [
                qoc_dag.get_node_end_time(node) for node in C_all_descendants
            ]
            C_l = (
                max(C_all_descendants_end_time) - qoc_dag.get_node_end_time(C)
                if len(C_all_descendants_end_time) != 0
                else base / 2
            )

            score = (
                base + B_prime_latency if B_prime_l >= C_l else base + C_l - B_prime_l
            )
    else:
        raise Exception(f"merge heuristic {merge_heuristic} has not been implemented.")
    return score
