"""
A python interface to use grami on the remote server.
"""
from collections import deque
from itertools import chain, combinations
from functools import reduce
import time
import uuid

from typing import Any, Callable, List, Dict

import networkx as nx
from networkx.algorithms import isomorphism as iso
from networkx.algorithms.isomorphism.vf2userfunc import DiGraphMatcher
from networkx.classes.digraph import DiGraph

from qiskit.dagcircuit import dagcircuit
from bidict import bidict

from sshtool import SSHTool
from gramigraph import GramiGraph


class PYGRAMI:
    def __init__(self, save_to_file: bool = False) -> None:
        self.save_to_file = save_to_file
        self.grami_graph: GramiGraph = GramiGraph()
        self.freq_subgraphs: List[Dict[str, Any]] = None

        # sshtool configs
        proxy_hostname = "ilab.cs.rutgers.edu"
        host_name = "seastar.cs.rutgers.edu"
        port = 22
        username = "yc827"
        password = "adq*ndu0ufr-ANM8mvh"

        self._ssht = SSHTool(
            (host_name, port),
            username,
            password,
            via=(proxy_hostname, port),
            via_user=username,
            via_auth=password,
        )

    def get_freq_subgraphs(self) -> List[GramiGraph]:
        return self.freq_subgraphs

    def run(
        self, circuit_dag: dagcircuit.DAGCircuit, support: int = 2, maxN: int = 3
    ) -> None:
        start = time.perf_counter()

        self.grami_graph = GramiGraph()
        self.grami_graph.from_circuit_dag(circuit_dag)
        grami_graph_string = self.grami_graph.generate_grami_graph_string()

        stop = time.perf_counter()
        elapsed_time = stop - start
        print(f"\tConvert dag to grami graph {elapsed_time:.4f}")

        if self.save_to_file:
            with open("grami_result/test_input.txt", "w") as fp:
                fp.write(grami_graph_string)

        # print(self.test_graph.split('\n')[:10])
        # nV = sum(1 for line in grami_graph_string.split('\n') if line.startswith('v'))
        # nE = sum(1 for line in grami_graph_string.split('\n') if line.startswith('e'))
        # logging.info('Generated grami input graph has %d nodes and %d edges.', nV, nE)
        graph_fname = str(uuid.uuid4())
        cmd = f"""
        cd /common/users/yc827/GraMi ;
        echo "{grami_graph_string}" > Datasets/{graph_fname}-input.txt ;
        """

        start = time.perf_counter()
        self._ssht.run(cmd)
        stop = time.perf_counter()
        elapsed_time = stop - start
        print(f"\tUpload input graph {elapsed_time:.4f}")

        # logging.info('Running GRAMI.')
        cmd = f"""
        cd /common/users/yc827/GraMi ;
        ./grami -f {graph_fname}-input.txt -s {support} -l 5 -t 1 -p 0 &> Output/{graph_fname}-output.txt;
        """

        start = time.perf_counter()
        self._ssht.run(cmd)
        stop = time.perf_counter()
        elapsed_time = stop - start
        print(f"\tRunning GRAMI. {elapsed_time:.4f}")

        # logging.info('Downloading grami outputs.')
        start = time.perf_counter()
        self._ssht.transfer(
            remote_path=f"/common/users/yc827/GraMi/Output/{graph_fname}-output.txt",
            local_path="grami_result/",
        )
        stop = time.perf_counter()
        elapsed_time = stop - start
        print(f"\tDownloading grami outputs. {elapsed_time:.4f}")

        # logging.info('Generating frequent subcircuits from grami output.')
        start = time.perf_counter()
        grami_output = open(f"grami_result/{graph_fname}-output.txt", mode="r").read()
        self.freq_subgraphs = self._get_freq_subgraphs_from_grami_output(
            self.grami_graph, grami_output, support, maxN
        )
        stop = time.perf_counter()
        elapsed_time = stop - start
        print(
            f"\tGenerating frequent subcircuits from grami output. {elapsed_time:.4f}"
        )

    def _get_freq_subgraphs_from_grami_output(
        self,
        grami_input_graph: GramiGraph,
        grami_output_string: str,
        support: int,
        maxN: int,
    ) -> List[Dict[str, Any]]:
        """
        Filter and generate the freqeuent patterns from the grami output.

        The current "grami output" contains all intermediate output from the original program as the
        frequency of a pattern is not stored for general grami output but only be printed in the
        middle of the execution.
        An example of the output file can be found under grami_result/test_output.txt.
        The common pattern looks like
        -------------------------Looking into frequency of: v 0 1
        v 1 1
        e 0 1 2

        called pruned lists
        called Automorphism pruned lists
        1
        0
        Freq: 8
        What we want to do is to get the part that has the frequency larger than the min support.
        Then generate the pattern in GramiGraph format (it provides from_circuit_dag/to_circuit_dag).
        At this stage, we do not consider the overlapping, priority of keeping among different
        patterns.
        """
        subgraph_lines = list()
        tmp_freq_subgraphs = []
        all_freq_subgraphs = []
        node_match = iso.categorical_node_match("label", "")
        edge_match = iso.categorical_edge_match("label", "")

        for line in grami_output_string.split("\n"):
            if line.startswith("Freq: "):
                freq = int(line.split(": ")[1])
                if freq < support:
                    continue

                freq_G = GramiGraph()
                grami_graph_string = "# t 1\n"
                grami_graph_string += "\n".join(
                    [
                        s_line
                        for s_line in subgraph_lines
                        if (s_line.startswith("v") or s_line.startswith("e"))
                    ]
                )
                try:
                    freq_G.from_grami_graph_string(
                        grami_graph_string, grami_input_graph
                    )
                except:
                    raise Exception(f"Invalid graph graph string {grami_graph_string}")

                # Filter out some invalid frequent subgraph
                nV, nE = freq_G.num_nodes, freq_G.num_edges
                if nE < nV - 1 or nV == 1:
                    continue

                # Any two nodes in the frequent subgraph have a path that includes edges
                # that are not in the frequent subgraph is invalid (cannot be merged).
                # for node1_idx, node2_idx in combinations(nx.topological_sort(freq_G.get_nx_graph()), 2)
                # NOTE This step might be heavy when the original graph is large.
                # FIXME This filter step seems redundant as it needs to find the isomorphic subgraph in
                # the original graph, which represents the original circuit. No matter what, we will have
                # to find the iso-subgraphs in the next step for potential merging.

                # 1. Find all (induced) isomorphic subgraph in the original graph.
                # 2. For all pair of nodes in the iso-subgraph, get all paths from the two nodes, if there
                # are edges in the path that is not in the iso-subgraph, then this subgraph will be marked
                # as invalid.
                # 3. If the number of valid subgraph is less than <support>, we skip this frequent subgraph.
                # FIXME This is wrong!!! but now it does not matter. Fixit later.
                def __contains_invalid_edges(
                    ori_nx_G: DiGraph,
                    freq_nx_G: DiGraph,
                    source: int,
                    target: int,
                    node_idx_dict: bidict,
                ) -> bool:
                    """
                    Given two nodes source and target in the original graph (nx.Digraph), find all simple paths
                    between the two nodes. If the path contains an edge that does not exist in the frequent
                    subgraph, then returns False. Otherwise returns True.
                    """
                    s_node = grami_input_graph.get_dag_node_from_index(source)
                    t_node = grami_input_graph.get_dag_node_from_index(target)
                    s_qidx = set(q.index for q in s_node.qargs)
                    t_qidx = set(q.index for q in t_node.qargs)
                    common_qidx = s_qidx & t_qidx

                    queue = deque([source])
                    visited = set([source])
                    returnFlag = False
                    while len(queue) > 0 and not returnFlag:
                        cur_node_idx = queue[0]
                        queue.popleft()
                        for _, ori_next_node_idx in ori_nx_G.out_edges(cur_node_idx):
                            if ori_next_node_idx in visited:
                                continue
                            next_node = grami_input_graph.get_dag_node_from_index(
                                ori_next_node_idx
                            )
                            next_node_qidx = set(q.index for q in next_node.qargs)
                            if len(next_node_qidx & common_qidx) == 0:
                                continue
                            # TODO add an assert len(new_next_node_idx) == 1
                            if ori_next_node_idx not in node_idx_dict.inverse:
                                return True
                            else:
                                if ori_next_node_idx == target:
                                    returnFlag = True
                                visited.add(ori_next_node_idx)
                                queue.append(ori_next_node_idx)

                    return False

                def __iso_pair_is_valid(
                    ori_nx_G: DiGraph, freq_nx_G: DiGraph, iso_pair
                ) -> bool:
                    """
                    Given a matching(iso_pair) from the subgraph isomorphism between the original graph and
                    the frequent subgraph. If there are no "invalid" edge between any two nodes in the frequent
                    subgraph, then returns True. Otherwise returns False.
                    """
                    node_idx_dict = bidict()
                    for ori_nid, new_nid in iso_pair.items():
                        node_idx_dict[new_nid] = ori_nid
                    freq_G_topo_nodes = nx.topological_sort(freq_nx_G)
                    ori_node_idx = [
                        node_idx_dict[new_nid] for new_nid in freq_G_topo_nodes
                    ]
                    for node1_idx, node2_idx in combinations(ori_node_idx, 2):
                        if __contains_invalid_edges(
                            ori_nx_G, freq_nx_G, node1_idx, node2_idx, node_idx_dict
                        ):
                            return False
                    return True

                def __count_valid_iso_subgraph(
                    ori_nx_G: DiGraph,
                    freq_nx_G: DiGraph,
                    node_match: Callable,
                    edge_match: Callable,
                ) -> int:
                    gm = DiGraphMatcher(
                        G1=ori_nx_G,
                        G2=freq_nx_G,
                        node_match=node_match,
                        edge_match=edge_match,
                    )
                    valid_count = sum(
                        1
                        for iso_pair in gm.subgraph_isomorphisms_iter()
                        if __iso_pair_is_valid(ori_nx_G, freq_nx_G, iso_pair)
                    )

                    return valid_count

                # freq_G.draw_grami_graph()
                valid_subgraph_number = __count_valid_iso_subgraph(
                    ori_nx_G=grami_input_graph.get_nx_graph(),
                    freq_nx_G=freq_G.get_nx_graph(),
                    node_match=node_match,
                    edge_match=edge_match,
                )
                if valid_subgraph_number < support:
                    continue

                nQ = len(
                    reduce(
                        lambda x, y: x | y,
                        [
                            set(freq_G.get_dag_node_from_index(n).qargs)
                            for n in range(freq_G.num_nodes)
                        ],
                    )
                )
                # details = [set(freq_G.get_dag_node_from_index(n).qargs)
                #             for n in range(freq_G.num_nodes)]
                if nQ > maxN:
                    continue

                tmp_freq_subgraphs.append(
                    {
                        "graph": freq_G,
                        "nV": nV,
                        "nE": nE,
                        "nQ": nQ,
                        # 'details': details,
                        "freq": valid_subgraph_number,
                    }
                )

            elif line.startswith("---"):
                subgraph_lines = [line.split(": ")[1].strip()]
            else:
                subgraph_lines.append(line.strip())

        # Sort all frequent subgraph by # nodes in decreasing order.
        # If a frequent subgraph is a subgraph isomorphism of another frequent subgraph with less or equal
        # appearance, then this subgraph should be skiped.
        def __subgraph_is_covered_by_another(
            freq_subgraph: Dict[str, Any],
            all_freq_subgraphs: List[Dict[str, Any]],
            node_match: Callable,
            edge_match: Callable,
        ) -> bool:
            freq_G, nV, nE, freq = (
                freq_subgraph["graph"],
                freq_subgraph["nV"],
                freq_subgraph["nE"],
                freq_subgraph["freq"],
            )
            for graph in all_freq_subgraphs:
                if nV > graph["nV"] or nE > graph["nE"]:
                    continue
                valid_subgraph_number = __count_valid_iso_subgraph(
                    ori_nx_G=graph["graph"].get_nx_graph(),
                    freq_nx_G=freq_G.get_nx_graph(),
                    node_match=node_match,
                    edge_match=edge_match,
                )
                # FIXME: A frequent subgraph with more nodes could have multiple appearance of another subgraph.
                if valid_subgraph_number >= 1 and graph["freq"] == freq:
                    return True
            return False

        tmp_freq_subgraphs.sort(key=lambda x: x["nV"], reverse=True)
        for freq_subgraph in tmp_freq_subgraphs:
            # freq_subgraph['graph'].print_grami_graph()
            if __subgraph_is_covered_by_another(
                freq_subgraph=freq_subgraph,
                all_freq_subgraphs=all_freq_subgraphs,
                node_match=node_match,
                edge_match=edge_match,
            ):
                continue
            all_freq_subgraphs.append(freq_subgraph)

        return all_freq_subgraphs

    # def dag_to_digraph(self) -> DiGraph:

    #     dag_digraph = DiGraph()

    #     node_idx = dict()
    #     # write vertices in the format 'v [index] [label: gate_name]'
    #     for index, node in enumerate(self.dag.topological_op_nodes()):
    #         node_idx[node] = index
    #         label = get_node_name_for_digraph(node)
    #         # name = node.name if node.name != 'u' else ''.join(format(x, "10.3f") for x in node._op._params)
    #         # label = node.name
    #         if label not in self.node_label:
    #             self.node_label[label] = len(self.node_label)
    #         dag_digraph.add_node(index, label=label, qargs = node.qargs, id = node._node_id, group = [])
    #     # End for

    #     # write edge in the format 'e [s_index] [t_index] [label: gate connection type]
    #     for node in self.dag.topological_op_nodes():
    #         s_index = node_idx[node]
    #         s_qidx = [q.index for q in node.qargs]

    #         for succ_node in self.dag.successors(node):
    #             if succ_node.type != 'op':
    #                 continue
    #             # For each succ_node, we want to record which qubit(s) it share with the current node.
    #             common_qidx = [edge_data.index
    #                            for _, s_node, edge_data in self.dag.edges(nodes=node)
    #                            if s_node == succ_node]

    #             t_index = node_idx[succ_node]
    #             t_qidx = [q.index for q in succ_node.qargs]

    #             label = '.'.join([str(sorted(s_qidx).index(qidx)) for qidx in s_qidx if qidx in common_qidx])
    #             label += '-'
    #             label += '.'.join([str(sorted(t_qidx).index(qidx)) for qidx in t_qidx if qidx in common_qidx])

    #             if label not in self.edge_label:
    #                 self.edge_label[label] = len(self.edge_label)
    #             dag_digraph.add_edge(s_index, t_index, common=label)
    #         # End for
    #     # End for

    #     return dag_digraph
