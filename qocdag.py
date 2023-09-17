from collections import Counter, deque
from copy import deepcopy
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode
from qiskit.converters import dag_to_circuit

import retworkx as rx

from helper import get_qreg_index, nodes2circ
from pulseDB import PulseDB


class QOCDAG:
    def __init__(self, pulse_db: PulseDB, dag: DAGCircuit, qregs: QuantumRegister):
        self.dag = deepcopy(dag)
        self.qregs = deepcopy(qregs)
        self.pulse_db: PulseDB = pulse_db

        self.c_path: List[DAGNode] = []
        self.node_end_time: Dict[DAGNode, float] = dict()

        self.latency: float = 0.0
        self.status_changed: bool = True

    def get_quantum_circuit(self) -> QuantumCircuit:
        return dag_to_circuit(self.dag)

    def get_dag_circuit(self) -> DAGCircuit:
        return self.dag

    def has_node(self, node: DAGNode) -> bool:
        return node in self.dag.nodes()

    def get_node_end_time(self, node: DAGNode) -> float:
        if isinstance(node, DAGOpNode):
            raise Exception("Node type is not op.")
        if not self.has_node(node):
            raise Exception(f"Dag does not have node {node.name}")
        if self.status_changed:
            self.get_critical_path()

        return self.node_end_time[node]

    def count_gates_details(self) -> Counter:
        count = Counter()
        for node in self.dag.topological_op_nodes():
            count[node.name] += 1
        return count

    def get_critical_path(
        self, print_details: bool = False
    ) -> Tuple[List[DAGNode], float]:
        if not self.status_changed:
            return self.c_path, self.latency

        dist = {}
        for i, node in enumerate(self.dag.topological_op_nodes()):
            pred_nodes = [
                pred_node
                for pred_node in self.dag.predecessors(node)
                if isinstance(pred_node, DAGOpNode)
            ]
            if print_details:
                print(f"{i:5}/{len(list(self.dag.topological_op_nodes())):5}", end="\r")
            node_latency, _ = self.pulse_db.get_optimal_pulse_time([node])
            us = [
                (dist[pred_node][0] + node_latency, pred_node)
                for pred_node in pred_nodes
            ]
            if len(us) == 0:
                dist[node] = (node_latency, node)
            else:
                dist[node] = max(us, key=lambda x: x[0])
            self.node_end_time[node] = dist[node][0]

        u = None
        v = max(dist, key=lambda x: dist[x][0])
        l = dist[v][0]
        path = []
        while u != v:
            path.append(v)
            u = v
            v = dist[v][1]
        path.reverse()

        self.status_changed = False
        self.c_path, self.latency = path, l
        return self.c_path, self.latency

    # Copy from qiskit
    def _collect_1q_runs(self) -> List[List[DAGNode]]:
        """Return a set of non-conditional runs of 1q "op" nodes."""

        def filter_fn(node: DAGNode):
            return (
                isinstance(node, DAGOpNode)
                and len(node.qargs) == 1
                and len(node.cargs) == 0
                and node.op.condition is None
                and not node.op.is_parameterized()
                and isinstance(node.op, Gate)
                # FIXME current new node does not have this attr
                # and hasattr(node.op, "__array__")
            )

        return rx.collect_runs(self.dag._multi_graph, filter_fn)

    def _collect_2q_runs(self) -> List[List[DAGNode]]:
        from qiskit.transpiler.passes import Collect2qBlocks

        collect_run = Collect2qBlocks()
        collect_run.run(self.dag)
        cont_2q_nodes = collect_run.property_set["block_list"]

        return cont_2q_nodes

    def _collect_3q_runs(self) -> List[List[DAGNode]]:
        merged = set()
        cont_3q_nodes = []
        topo_order_nodes = list(self.dag.topological_op_nodes())
        for node in topo_order_nodes:
            if node in merged:
                continue
            for succ in self.dag.successors(node):
                if (
                    not isinstance(succ, DAGOpNode)
                    or succ in merged
                    or not self.check_can_merge([node, succ])
                ):
                    continue
                new_nodes = self.generate_extended_nodes([node, succ], merged)
                assert len(new_nodes) >= 2
                cont_3q_nodes.append(new_nodes)
                merged |= set(new_nodes)
                break
        for node in topo_order_nodes:
            if node not in merged:
                cont_3q_nodes.append([node])
        return cont_3q_nodes

    def collect_nq_runs(self, n: int) -> List[List[DAGNode]]:
        if n == 1:
            return self._collect_1q_runs()
        elif n == 2:
            return self._collect_2q_runs()
        elif n == 3:
            return self._collect_3q_runs()
        else:
            raise Exception(f"Undefined collect_{n}q_runs().")

    # TODO: find an efficient way to check
    def check_can_merge(self, nodes: List[DAGNode]) -> bool:
        # node_depths = [self.get_dag_node_depth(node) for node in nodes]
        # node_depths.sort()
        # if all(d2 - d1 == 1
        #        for d1, d2 in
        #        zip(node_depths[:-1], node_depths[1:])):
        #     return True
        # else:
        #     return False

        all_descendants, all_ancestors = set(), set()
        node_idxes = [node._node_id for node in nodes]
        for node in nodes:
            all_descendants |= rx.descendants(self.dag._multi_graph, node._node_id)
        for node in nodes:
            all_ancestors |= rx.ancestors(self.dag._multi_graph, node._node_id)
            # early check
            if len((all_ancestors & all_descendants) - set(node_idxes)) != 0:
                return False
        return True

    def generate_extended_nodes(
        self, nodes: List[DAGNode], pattern_nodes: Set[DAGNode] = {}
    ):
        def _add_covered_nodes(
            all_covered: List[DAGNode], pred: bool = False
        ) -> Optional[List[DAGNode]]:
            covered_qidx = get_qreg_index(nodes)
            # FIXME the following
            # Here frontier will include all nodes in all_covered
            frontier = nodes.copy()
            while len(frontier) > 0:
                node = frontier[0]
                frontier = frontier[1:]

                all_next_nodes = (
                    self.dag.predecessors(node)
                    if pred is True
                    else self.dag.successors(node)
                )
                all_next_nodes = [
                    node for node in all_next_nodes if isinstance(node, DAGOpNode)
                ]
                for next_node in all_next_nodes:
                    next_node_qidx = get_qreg_index([next_node])
                    if (
                        next_node in all_covered
                        or next_node in pattern_nodes
                        or len(next_node_qidx - covered_qidx) != 0
                        or (
                            pred and not self.check_can_merge([next_node] + all_covered)
                        )
                        or (
                            not pred
                            and not self.check_can_merge(all_covered + [next_node])
                        )
                    ):
                        continue
                    frontier.append(next_node)
                    if pred is True:
                        all_covered = [next_node] + all_covered
                    else:
                        all_covered.append(next_node)
            return all_covered

        if not self.check_can_merge(nodes):
            return None
        # Why need return
        # FIXME
        all_covered = nodes
        all_covered = _add_covered_nodes(all_covered, pred=False)
        all_covered = _add_covered_nodes(all_covered, pred=True)

        return all_covered

    def merge_dag_nodes(self, nodes: List[DAGNode]) -> Optional[DAGNode]:
        for node in nodes:
            if node not in self.dag.nodes():
                return None
        if len(nodes) <= 1:
            return None

        # Generate the subcircuit.
        idx_map, sub_circ, new_op, new_name = nodes2circ(nodes)
        # print(sub_circ.draw())
        new_qargs = [Qubit(self.qregs, idx_map[i]) for i in range(len(idx_map))]
        new_node = DAGOpNode(op=new_op, qargs=new_qargs, cargs=[])
        new_node.name = new_name
        new_nid = self._replace_nodes_with_a_new_node(nodes, new_node)

        self.status_changed = True

        return new_node

    def _replace_nodes_with_a_new_node(
        self, nodes: List[DAGNode], new_node: DAGNode
    ) -> int:
        new_nid = self.dag._multi_graph.add_node(new_node)
        new_node._node_id = new_nid

        nodes_nid = [node._node_id for node in nodes]

        pred_edges, succ_edges = list(), list()
        for node in nodes:
            pred_edges += [e for e in self.dag._multi_graph.in_edges(node._node_id)]
            succ_edges += [e for e in self.dag._multi_graph.out_edges(node._node_id)]
        pred_edges = [(e[0], e[1], e[2]) for e in pred_edges if e[0] not in nodes_nid]
        succ_edges = [(e[0], e[1], e[2]) for e in succ_edges if e[1] not in nodes_nid]

        # add new edges to dag
        # TODO use add_edges_from
        for pred, _, attr in pred_edges:
            self.dag._multi_graph.add_edge(pred, new_node._node_id, attr)
        for _, succ, attr in succ_edges:
            self.dag._multi_graph.add_edge(new_node._node_id, succ, attr)

        # remove old nodes from dag
        for node in nodes:
            self.dag._multi_graph.remove_node(node._node_id)

        return new_nid
