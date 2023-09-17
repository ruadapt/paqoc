"""
Object to represent a quantum circuit as a directed labeled (both node and edge) graph.

The graph should be a valid Grami input. The nodes in the graph should represent quantum
gates while the edges correspond to the connection between quantum gates. A directed edge 
from node A to node B means some qubit(s) pass from gate A to gate B.
"""
from collections import defaultdict
import sys
from typing import Dict, List
from matplotlib import pyplot as plt
from qiskit.dagcircuit import dagcircuit, dagnode
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode
import networkx as nx

from bidict import bidict


class GramiGraph:
    def __init__(self):
        self.grami_graph = nx.DiGraph()
        # Each node in the grami graph is represented as an integer.
        # grami_node_idx is a dictionary that returns the index of
        # a dag node in the grami graph.
        self.grami_node_idx: bidict[dagnode.DAGNode, int] = bidict()
        # Each node/edge label in the grami graph is an integer.
        # Similar to grami_node_idx, we use a dictionary to store
        # the mapping.
        self.grami_node_label_idx: bidict[str, int] = bidict()
        self.grami_edge_label_idx: bidict[str, int] = bidict()
        # Each node label corresponds to a few dag nodes
        self.grami_node_label_dag_node: Dict[str, List[dagnode.DAGNode]] = defaultdict(
            list
        )
        self.num_nodes = 0
        self.num_edges = 0

    def add_node_with_label(self, dag_node: dagnode.DAGNode, label: str) -> None:
        index = len(self.grami_node_idx)
        self.grami_node_idx[dag_node] = index
        if label not in self.grami_node_label_idx:
            self.grami_node_label_idx[label] = len(self.grami_node_label_idx)
        self.grami_graph.add_node(index, dag_node=dag_node, label=label)
        self.grami_node_label_dag_node[label].append(dag_node)
        self.num_nodes += 1
        assert (
            self.num_nodes == self.grami_graph.number_of_nodes()
        ), "Number of nodes do not match"

    def add_node(self, dag_node: dagnode.DAGNode) -> None:
        from helper import generate_node_label

        label = generate_node_label(dag_node)
        self.add_node_with_label(dag_node, label)

    def add_nodes(self, nodes: List[dagnode.DAGNode]) -> None:
        [self.add_node(node) for node in nodes]

    def add_edge_with_label(self, u_idx: int, v_idx: int, label: str) -> None:
        assert self.grami_graph.has_node(u_idx), f"Node {u_idx} does not exist."
        assert self.grami_graph.has_node(v_idx), f"Node {v_idx} does not exist."

        if label not in self.grami_edge_label_idx:
            self.grami_edge_label_idx[label] = len(self.grami_edge_label_idx)
        self.grami_graph.add_edge(u_idx, v_idx, label=label)
        self.num_edges += 1
        assert (
            self.num_edges == self.grami_graph.number_of_edges()
        ), "Number of edges do no match."

    def add_edge(
        self, circuit_dag: dagcircuit.DAGCircuit, u: dagnode.DAGNode, v: dagnode.DAGNode
    ) -> None:
        from helper import generate_edge_label

        label = generate_edge_label(circuit_dag, u, v)
        assert u in self.grami_node_idx, f"Node {u.name} does not exist."
        assert v in self.grami_node_idx, f"Node {v.name} does not exist."

        u_idx, v_idx = self.grami_node_idx[u], self.grami_node_idx[v]
        self.add_edge_with_label(u_idx, v_idx, label)

    def add_edges(
        self, circuit_dag: dagcircuit.DAGCircuit, edges: List[List[dagnode.DAGNode]]
    ) -> None:
        [self.add_edge(circuit_dag, u, v) for u, v in edges]

    def get_nx_graph(self) -> nx.DiGraph:
        return self.grami_graph

    def get_dag_node_from_index(self, index: int) -> dagnode.DAGNode:
        dag_node = self.grami_node_idx.inverse[index]
        assert len(dag_node) == 1, f"Invalid bidict grami_node_idx with val {index}."
        return dag_node[0]

    def get_node_label_from_index(self, index: int) -> str:
        label = self.grami_node_label_idx.inverse[index]
        assert len(label) == 1, f"Invalid bidict grami_node_label_idx with val {index}."
        return label[0]

    def get_edge_label_from_index(self, index: int) -> str:
        label = self.grami_edge_label_idx.inverse[index]
        assert len(label) == 1, f"Invalid bidict grami_edge_label_idx with val {index}."
        return label[0]

    def get_dag_nodes_from_label(self, label: str) -> List[dagnode.DAGNode]:
        return self.grami_node_label_dag_node[label]

    def generate_grami_graph_string(self) -> str:
        grami_graph_string: str = "# t 1\n"
        for node_index, node_data in self.grami_graph.nodes(data=True):
            grami_graph_string += (
                f"v {node_index} {self.grami_node_label_idx[node_data['label']]}\n"
            )
        for node1_indx, node2_index, edge_data in self.grami_graph.edges(data=True):
            grami_graph_string += f"e {node1_indx} {node2_index} {self.grami_edge_label_idx[edge_data['label']]}\n"
        return grami_graph_string.rstrip()

    def from_circuit_dag(self, circuit_dag: dagcircuit.DAGCircuit):
        """
        Take in a dagcircuit from qiskit and construct the grami graph object.
        """
        self.add_nodes(list(circuit_dag.topological_op_nodes()))

        # Generate edges
        dag_node: dagnode.DAGNode
        for dag_node in circuit_dag.topological_op_nodes():
            for succ_dag_node in circuit_dag.successors(dag_node):
                if not isinstance(succ_dag_node, DAGOpNode):
                    continue
                self.add_edge(circuit_dag, dag_node, succ_dag_node)

    def from_grami_graph_string(self, grami_graph_string: str, grami_input_graph):
        """
        Take in a grami graph string, together with the corresponding dictionary for nodes, node
        labels and edge labels and construct the grami graph object.
        """
        for i, line in enumerate(grami_graph_string.split("\n")[1:], 1):
            # First line should be '# t 1' which should be skipped.
            if line.startswith("v"):
                words = line.strip().split(" ")
                assert (
                    len(words) == 3
                ), f"Invalid input grami graph string at line {i} {line}"
                idx, label_idx = int(words[1]), int(words[2])
                label = grami_input_graph.get_node_label_from_index(label_idx)
                # dag_node = grami_input_graph.get_dag_node_from_index(idx)
                # NOTE This is a workaround as we don't map the node in the frequent graph to
                # any dag node in the original graph.
                for dag_node in grami_input_graph.get_dag_nodes_from_label(label):
                    if dag_node not in self.grami_node_idx:
                        break
                self.add_node_with_label(dag_node, label)
            elif line.startswith("e"):
                words = line.strip().split(" ")
                assert (
                    len(words) == 4
                ), f"Invalid input grami graph string at line {i} {line}"
                u_idx, v_idx, label_idx = [int(n) for n in words[1:]]
                label = grami_input_graph.get_edge_label_from_index(label_idx)
                self.add_edge_with_label(u_idx, v_idx, label)
            else:
                raise Exception(
                    "Invalid input grami graph string at line " + {i}, {line}
                )

    def draw_grami_graph(self):
        plt.figure(figsize=(16, 12))
        G = self.grami_graph
        pos = nx.nx_pydot.graphviz_layout(G)
        node_labels = nx.get_node_attributes(G, "label")
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            labels=node_labels,
            node_size=1e3,
            font_size=20,
        )
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=18)

    def print_grami_graph(self):
        print("nodes:")
        for node in self.grami_graph.nodes(data="label"):
            print("\t", node)
        print("edges:")
        for edge in self.grami_graph.edges(data="label"):
            print("\t", edge)
