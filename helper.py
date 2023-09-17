from copy import deepcopy
import hashlib
from os import system
from typing import Any, Dict, List, Set, Tuple
from itertools import product
from functools import reduce
import numpy as np
import ast
import sys

import networkx as nx
from networkx.algorithms import isomorphism as iso
from networkx.algorithms.isomorphism.vf2userfunc import DiGraphMatcher
from qiskit import dagcircuit

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.dagcircuit import dagnode
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info.operators import Operator
from gramigraph import GramiGraph

from bidict import bidict


def generate_2d_grid_coupling(N: int) -> List[List[int]]:
    graph = []
    for x, y in product(range(N), range(N)):
        i = y * N + x
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            _x, _y = x + dx, y + dy
            if _x < 0 or _x >= N or _y < 0 or _y >= N:
                continue
            j = _y * N + _x
            graph.append([i, j])
    graph.sort()
    return graph


def get_qreg_index(nodes: List[DAGNode]) -> Set[int]:
    return reduce(
        lambda x, y: x | y, [set([q.index for q in node.qargs]) for node in nodes]
    )


def get_layers(dag: DAGCircuit) -> List[List[DAGNode]]:
    graph_layers = dag.multigraph_layers()
    try:
        next(graph_layers)  # Remove input nodes
    except StopIteration:
        return
    layers = []
    for graph_layer in graph_layers:
        op_nodes = [node for node in graph_layer if isinstance(node, DAGOpNode)]
        op_nodes.sort(
            key=lambda nd: nd.name
            if nd.name != "u"
            else "".join(format(x, "10.3f") for x in nd._op._params)
        )
        layers.append(op_nodes)

    return layers


def get_node_name_for_qoc(node: DAGNode) -> str:
    # name = node.name
    name = (
        node.name
        if node.name != "u"
        else "".join(format(x, "10.3f") for x in node._op._params)
    )
    return name


def str2nparray(array_string):
    array_string = ",".join(array_string.replace("[ ", "[").split())
    return np.array(ast.literal_eval(array_string))


def c_to_r_mat(M):
    return np.asarray(np.bmat([M.real, M.imag]))


# FIXME: Looks like rotation degree will be lost
def nodes2circ(nodes: List[DAGNode]) -> Tuple[bidict, QuantumCircuit, Gate, str]:
    """Convert a list of connected dag nodes to the circuit"""
    all_idxes = get_qreg_index(nodes)
    idx_map = dict((idx, i) for i, idx in enumerate(all_idxes))
    N = len(all_idxes)

    qr = QuantumRegister(N, "q")
    circ = QuantumCircuit(N)
    for node in nodes:
        new_qargs = [qr[idx_map[q.index]] for q in node.qargs]
        circ._append(node.op, new_qargs, [])

    layers = get_layers(circuit_to_dag(circ))

    qr = QuantumRegister(N, "q")
    circ = QuantumCircuit(N)

    new_idx_map = dict()
    new_name = []
    for layer in layers:
        for node in layer:
            new_name.append(get_node_name_for_qoc(node))
            for idx in [q.index for q in node.qargs]:
                if idx not in new_idx_map:
                    new_idx_map[idx] = len(new_idx_map)
            new_qargs = [qr[new_idx_map[q.index]] for q in node.qargs]
            circ._append(node.op, new_qargs, [])

    new_name = "_".join(new_name)
    # Temp walkaround, FIXME now names could be same for different unitary.
    U = Operator(circ).data
    _, circ_key_hash = unitary2circ_key(U)
    new_name = new_name + "." + circ_key_hash[:5]
    new_op = circ.to_gate()
    new_op.name = new_name

    ret_idx_map = bidict({new_idx_map[idx_map[idx]]: idx for idx in all_idxes})

    return ret_idx_map, circ, new_op, new_name


def merge_freq_subgraphs(
    graphs: List[Dict[str, Any]], allowed_count: int, qoc_dag
) -> List[Tuple[Dict[str, Any], List[List[DAGNode]]]]:
    from qocdag import QOCDAG

    qoc_dag: QOCDAG
    node_match = iso.categorical_node_match("label", "")
    edge_match = iso.categorical_edge_match("label", "")
    merged_lst = list()

    merged_count = 0
    for graph in graphs:
        freq_G: GramiGraph = graph["graph"]
        ori_grami_graph = GramiGraph()
        # print([node.name for node in
        #        qoc_dag.get_dag_circuit().topological_op_nodes()])

        ori_grami_graph.from_circuit_dag(qoc_dag.get_dag_circuit())
        ori_nx_G = ori_grami_graph.get_nx_graph()
        freq_nx_G = freq_G.get_nx_graph()
        freq_G_topo_nodes = list(nx.topological_sort(freq_nx_G))
        gm = DiGraphMatcher(
            G1=ori_nx_G, G2=freq_nx_G, node_match=node_match, edge_match=edge_match
        )
        # Since we keep merging gates without updating the ori_grami_graph, which is built
        # from an 'old' qoc_dag, ori_nodes may contain some dagnodes that have been merged.
        merged_nodes: List[DAGNode] = []
        for iso_pair in gm.subgraph_isomorphisms_iter():
            node_idx_dict = bidict()
            for ori_nid, new_nid in iso_pair.items():
                node_idx_dict[new_nid] = ori_nid
            ori_node_idx = [node_idx_dict[new_nid] for new_nid in freq_G_topo_nodes]
            ori_nodes = [
                ori_grami_graph.get_dag_node_from_index(ori_nid)
                for ori_nid in ori_node_idx
            ]
            # flag1 = qoc_dag.check_can_merge(ori_nodes)
            # prev_qoc_dag = deepcopy(qoc_dag)
            if qoc_dag.check_can_merge(ori_nodes):
                new_node = qoc_dag.merge_dag_nodes(ori_nodes)
                if new_node is not None:
                    merged_nodes.append(new_node)
        if len(merged_nodes) > 0:
            merged_lst.append((graph, merged_nodes))
            merged_count += 1
            graph["graph"].print_grami_graph()
        if merged_count == allowed_count:
            break

    return merged_lst


def generate_node_label(node: dagnode.DAGNode) -> str:
    """
    Return the label of the dag node.
    """
    name = node.name
    # Use the following code if we want to consider the rotation degree as parts of the node name.
    # name = node.name if node.name != 'u' else ''.join(format(x, "10.3f") for x in node._op._params)
    return name


def generate_edge_label(
    circuit_dag: dagcircuit.DAGCircuit, node1: dagnode.DAGNode, node2: dagnode.DAGNode
) -> str:
    """
    Return the label of the edge between two dag nodes.
    """
    # s_index =  self.grami_node_idx[node1]
    s_qidx = [q.index for q in node1.qargs]
    # t_index =  self.grami_node_idx[node2]
    t_qidx = [q.index for q in node2.qargs]
    # For each succ_node, we want to record which qubit(s) it share with the current node.
    # FIXME add checking for node1 and node2 in the circuit dag.

    # We need to find shared qubits between node1 and node2.
    # Note that even the same qubits are shared by the two nodes,
    # this does not mean they are directly connected through
    # the corresponding qubits.
    # FIXME: does "circuit_dag.edges(nodes=node1)" work properly? as node1 is dagnode not node index.
    # FIXME: should be qiskit version check
    from ppretty import ppretty

    if sys.version_info.minor > 7:
        #     for _, s_node, edge_data in circuit_dag.edges(nodes=node1):
        #         if s_node == node2:
        #             print(ppretty(node1))
        #             print(ppretty(node2))
        #             print(edge_data.index)
        #     common_qidx = [
        #         edge_data["wire"].index
        #         for _, s_node, edge_data in circuit_dag.edges(nodes=node1)
        #         if s_node == node2
        #     ]
        # else:
        common_qidx = [
            edge_data.index
            for _, s_node, edge_data in circuit_dag.edges(nodes=node1)
            if s_node == node2
        ]
    all_qidx_dict = {qidx: [] for qidx in common_qidx}
    [
        all_qidx_dict[qidx].append(i)
        for i, qidx in enumerate(s_qidx)
        if qidx in all_qidx_dict
    ]
    [
        all_qidx_dict[qidx].append(i)
        for i, qidx in enumerate(t_qidx)
        if qidx in all_qidx_dict
    ]

    label = "+".join(
        [
            ".".join([str(i) for i in i_lst])
            for qidx, i_lst in all_qidx_dict.items()
            if len(i_lst) == 2
        ]
    )

    # label = '.'.join([str(sorted(s_qidx).index(qidx))
    #                  for qidx in s_qidx if qidx in common_qidx])
    # label += '-'
    # label += '.'.join([str(sorted(t_qidx).index(qidx))
    #                   for qidx in t_qidx if qidx in common_qidx])
    return label


def unitary2circ_key(U):
    U = np.round(c_to_r_mat(U), 2)
    U += 0.0
    circ_key = str(U)
    circ_key_hash = hashlib.sha256(circ_key.encode("utf-8")).hexdigest()
    return circ_key, circ_key_hash
