from collections import defaultdict
import time
from typing import Any, Dict, List, Set, Tuple
from numpy.lib import math
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGNode

from pulseDB import PulseDB
from qocdag import QOCDAG

import pickle

class PAQOC:
    def __init__(
        self, 
        name: str,
        circ: QuantumCircuit, 
        pulse_db_filename: str = 'PulseDB/pulse_db.csv',
        min_freq: int = 2, # the minimum appearance of the pattern in the circuit
        prune_method: str = 'random', 
        merge_heuristic: str = 'random',
        topk: int = 5, # merge topk candidates at each iteration
        maxN: int = 3, # maximum size of the merged gate
        maxG: int = math.inf, # maximum number of patterns
        proctect_pattern: bool = False, # are we allowed to touch pattern during merging
        merge_cont1q: bool = True, # merge continous 1q gates at the very beginning
        merge_cont2q: bool = True, # merge continous 2q gates at the very beginning
        enable_grami: bool = True, # use grammi to find patterns
        enable_pamerge: bool = True # use pamerge to further merge gates
        ) -> None:
        
        self.circ_name: str = name
        self.original_circ: QuantumCircuit = circ.copy()
        self.original_dag: DAGCircuit = circuit_to_dag(circ)
        self.pulse_db: PulseDB = PulseDB(pulse_db_filename)
        self.qoc_dag: QOCDAG = QOCDAG(pulse_db=self.pulse_db,
                                      dag=self.original_dag, 
                                      qregs=self.original_circ.qregs[0])
        self.pattern_nodes: Set[DAGNode] = set()
        
        self.min_freq = min_freq
        self.prune_method = prune_method
        self.merge_heuristic = merge_heuristic
        self.topk = topk
        self.maxG = maxG
        self.maxN = maxN
        self.proctect_pattern = proctect_pattern
        self.merge_cont1q = merge_cont1q
        self.merge_cont2q = merge_cont2q
        self.enable_grami = enable_grami
        self.enable_pamerge = enable_pamerge
        
        self.c_path: List[DAGNode] = []
        self.latency: float = 0.0
        
        self.result: Dict[str, Any] = dict()
        
    def _calculate_compile_time(self) -> float:
        total_compile_time = 0
        if self.enable_grami:
            total_compile_time += sum(self.result['grami']['grami_time'])
        if self.enable_pamerge:
            if self.merge_cont1q:
                total_compile_time += self.result['merge1q']['time']
            if self.merge_cont2q:
                total_compile_time += self.result['merge2q']['time']
        
            total_compile_time += sum(self.result['pamerge']['get_candidates_time'])
            total_compile_time += sum(self.result['pamerge']['select_candidates_time'])
            total_compile_time += sum(self.result['pamerge']['merge_candidates_time'])
        
        total_compile_time += self.result['prestored_compile_time']
        
        return total_compile_time
        
        
        
    def get_quantum_circuit(self) -> QuantumCircuit:
        return self.qoc_dag.get_quantum_circuit()
    
    
    def get_critical_path(self) -> Tuple[List[DAGNode], float]:
        self.c_path, self.latency = self.qoc_dag.get_critical_path()
        return self.c_path, self.latency
    
    def print_critical_path(self, update: bool=False):
        if update:
            self.get_critical_path()
        for node in self.c_path:
            print(f'[{node.name}, {self.pulse_db.get_optimal_pulse_time([node])[0]}]', end='-')
        print()
        print(f'Total latency: {self.latency}.')
    
    def update_pattern_nodes(self, nodes: List[DAGNode]):
        # remove merged patterns
        self.pattern_nodes = set(node 
                                 for node in self.pattern_nodes 
                                 if self.qoc_dag.has_node(node))
        
        # add new patterns
        for node in nodes:
            if self.qoc_dag.has_node(node):
                self.pattern_nodes.add(node)
        
        
        
    def run_grami(self):
        from pygrami import PYGRAMI
        from helper import merge_freq_subgraphs
        
        grami_result = defaultdict(list)
        mergednbr = 0
        while True:
            # print(self.get_quantum_circuit().draw(idle_wires=False))
            # print([node.name 
            #        for node in 
            #       self.qoc_dag.get_dag_circuit().topological_op_nodes()])
            
            grami_start = time.perf_counter()
            
            grami = PYGRAMI(save_to_file=False)
            grami.run(self.qoc_dag.get_dag_circuit(), support=self.min_freq, maxN=self.maxN)
            freq_subgraphs: List[Dict[str, Any]] = grami.get_freq_subgraphs()
            if len(freq_subgraphs) == 0:
                break

            # sort based on total # of gates covered
            # TODO different frequent pattern selections
            # freq_subgraphs.sort(key = lambda x: (x["nV"] * x["freq"], x["nE"]), reverse=True)
            
            # Merge single qubit pattern first. Then merge two qubit pattern. Last others
            # All in coverage
            freq_subgraphs.sort(
                key = 
                lambda x:
                    (-x['nQ'], x['nV'] * x['freq']) if x['nQ'] <= 2 else (-3, x['nV'] * x['freq']),
                reverse=True    
                )

            merged_counter = merge_freq_subgraphs(freq_subgraphs, self.maxG - mergednbr, self.qoc_dag)
            mergednbr += len(merged_counter)
            
            grami_result['nfreq_graphs'].append(len(merged_counter))
            grami_result['merged_nodes'].append([merged_nodes for _, merged_nodes in merged_counter])
            grami_result['gate_number'].append(len(list(self.qoc_dag.get_dag_circuit().topological_op_nodes())))
            
            for freq_subgraph, merged_nodes in merged_counter:
                # TODO need to fix. It is still possible to merge "pattern" appear once
                if len(merged_nodes) > 1:
                    self.update_pattern_nodes(merged_nodes)

                    # freq_subgraph['graph'].print_grami_graph()
                    # print('freq: ', len(merged_nodes))

            grami_stop = time.perf_counter()
            grami_time = grami_stop - grami_start
            grami_result['grami_time'].append(grami_time)
            grami_result['gate_count'].append(self.qoc_dag.count_gates_details())
                
            if (len(merged_counter) == 0 or 
                mergednbr >= self.maxG):
                break
            
        self.result['grami'] = grami_result
            
            
    def run_merge_nq(self, n: int):

        for nqubits in range(1, n + 1):
            mergenq_start = time.perf_counter()
            
            cont_nq_nodes = self.qoc_dag.collect_nq_runs(nqubits)
            for nodes in cont_nq_nodes:
                if self.proctect_pattern:
                    pattern_idx = [i for i, node in enumerate(nodes) 
                                if (node in self.pattern_nodes and 
                                    len(node.qargs) == nqubits)]
                    pattern_idx = [-1] + pattern_idx + [len(nodes)]
                    for l, r in zip(pattern_idx[:-1], pattern_idx[1:]):
                        if (l + 1 < r and 
                            max(len(node.qargs) 
                                for node in nodes[l + 1 : r]) == nqubits):
                            self.qoc_dag.merge_dag_nodes(nodes[l + 1 : r])
                else:
                    self.qoc_dag.merge_dag_nodes(nodes)
        
            mergenq_stop = time.perf_counter()
            mergenq_time = mergenq_stop - mergenq_start
            
            self.result['merge' + str(nqubits) + 'q'] = {
                'gate_count': self.qoc_dag.count_gates_details(),
                'time': mergenq_time
                }
    
    
    def run_pamerge(self):
        import pamerge as pm
        
        assert (self.proctect_pattern or (len(self.pattern_nodes) == 0), 
                (f'proctect_pattern is {self.proctect_pattern}.' + 
                 'The # of patterns is {len(self.pattern_nodes)}'))
        
        pm_result = defaultdict(list)
        
        iter = 0
        # self.pulse_db.reset_prestored_compile_time();
        while True:
            iter += 1
            
            get_candidates_start = time.perf_counter()
            
            merge_candidates = pm.get_merge_candidates(self.qoc_dag, 
                                                       self.prune_method,
                                                       self.maxN,
                                                       self.pattern_nodes)
            
            get_candidates_stop = time.perf_counter()
            get_candidates_time = get_candidates_stop - get_candidates_start
            pm_result['get_candidates_time'].append(get_candidates_time)
            
            print(f'iter: {iter}\n\t# candidates: {len(merge_candidates)}')
            ##################################################
            print(f'\tcalculate scores...', end='')
            
            select_candidates_start = time.perf_counter()
            
            selects = [(pm.cal_merge_score(self.qoc_dag, 
                                           candidate, 
                                           self.merge_heuristic), 
                        candidate) 
                    for candidate in merge_candidates]
            selects.sort(reverse=True)
            
            select_candidates_stop = time.perf_counter()
            select_candidates_time = select_candidates_stop - select_candidates_start
            pm_result['select_candidates_time'].append(select_candidates_time)
            
            pm_result['candidates'].append(selects)
            
            print('Finish\n')
            ##################################################
            
            merge_candidates_start = time.perf_counter()
            
            merged_count = 0
            selected_candidates = []
            for _, nodes in selects:
                print(f'\tMerge {merged_count + 1} candidates...')
                if self.qoc_dag.merge_dag_nodes(nodes) is not None:
                    selected_candidates.append(nodes)
                    merged_count += 1
                if merged_count == self.topk:
                    break
                
            merge_candidates_stop = time.perf_counter()
            merge_candidates_time = merge_candidates_stop - merge_candidates_start
            pm_result['merge_candidates_time'].append(merge_candidates_time)
            
            pm_result['selected'].append(selected_candidates)
            
            _, latency = self.qoc_dag.get_critical_path()
            pm_result['compile_time'].append(self.pulse_db.get_prestored_compile_time())
            pm_result['latency'].append(latency)
            pm_result['gate_count'].append(self.qoc_dag.count_gates_details())
                
            if merged_count == 0:
                break
        
        self.result['pamerge'] = pm_result
            
    def run(self):
        self.result['original_gate_count'] = self.qoc_dag.count_gates_details()
        self.result['original_latency'] = self.get_critical_path()[1]
        
        if self.enable_grami:
            self.run_grami()
            
        if self.enable_pamerge:
            if self.merge_cont1q and self.merge_cont2q:
                self.run_merge_nq(2)
            elif self.merge_cont1q:
                self.run_merge_nq(1)
                
            # Print patterns
            # print(set(node.name for node in self.pattern_nodes))
                
            self.run_pamerge()
        
        self.result['final_gate_count'] = self.qoc_dag.count_gates_details()
        self.result['final_latency'] = self.get_critical_path()[1]
        self.result['prestored_compile_time'] = self.pulse_db.get_prestored_compile_time()
        self.result['total_compile_time'] = self._calculate_compile_time()
        
        # self.print_result()
        
        
    def save_to_pickle(self, file_name: str):
        pickle_result = dict()
        pickle_result['min_freq'] = self.min_freq
        pickle_result['prune_method'] = self.prune_method
        pickle_result['merge_heuristic'] = self.merge_heuristic
        pickle_result['maxN'] = self.maxN
        pickle_result['maxG'] = self.maxG
        pickle_result['proctect_pattern'] = self.proctect_pattern
        pickle_result['merge_cont1q'] = self.merge_cont1q
        pickle_result['merge_cont2q'] = self.merge_cont2q
        pickle_result['enable_grami'] = self.enable_grami
        pickle_result['circ_name'] = self.circ_name
        pickle_result['nqubits'] = self.original_circ.num_qubits
        pickle_result['ngates'] = self.original_circ.size()
        pickle_result['circ_name'] = self.circ_name
        
        if self.enable_grami:
            grami_result = self.result['grami']
            pickle_result['grami_time'] = grami_result["grami_time"]
            pickle_result['grami_npatterns'] = grami_result["nfreq_graphs"]
            pickle_result['grami_gate_count'] = grami_result["gate_count"]
        if self.enable_pamerge:
            if self.merge_cont1q:
                merge_1q_result = self.result["merge1q"]
                pickle_result['merge1q_time'] = merge_1q_result["time"]
                pickle_result['merge1q_gate_count'] = merge_1q_result["gate_count"]
            if self.merge_cont1q:
                merge_2q_result = self.result["merge2q"]
                pickle_result['merge2q_time'] = merge_2q_result["time"]
                pickle_result['merge2q_gate_count'] = merge_2q_result["gate_count"]
            
            pamerge_result = self.result['pamerge']    
            pickle_result['pamerge_niter'] = len(pamerge_result['latency'])
            
            pickle_result['pamerge_getT'] = pamerge_result["get_candidates_time"]
            pickle_result['pamerge_selectT'] = pamerge_result["select_candidates_time"]
            pickle_result['pamerge_mergeT'] = pamerge_result["merge_candidates_time"]
            pickle_result['pamerge_ncandidates'] = [len(x) for x in pamerge_result["candidates"]]
            pickle_result['pamerge_nselected'] = [len(x) for x in pamerge_result["selected"]]
            pickle_result['pamerge_latency'] = pamerge_result["latency"]
            pickle_result['pamerge_gate_count'] = pamerge_result["gate_count"]
            pickle_result['pamerge_compile_time'] = pamerge_result['compile_time']
        
        pickle_result['original_latency'] = self.result["original_latency"]
        pickle_result['final_latency'] = self.result["final_latency"]
        pickle_result['original_gate_count'] = self.result["original_gate_count"]
        pickle_result['final_gate_count'] = self.result["final_gate_count"]
        pickle_result['compile'] = self.result["total_compile_time"]
        pickle_result['calcualted_fidelity'] = self.get_fidelity()

        with open(file_name, 'wb') as fp:
            pickle.dump(pickle_result, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    def print_result(self, 
                     details: bool=False, 
                     save_to_file: str=None):
        # config
        out_str = 'Config:\n'
        out_str += f'Grami: min_freq: {self.min_freq}\n'
        out_str += f'PAQOC: prune_method: {self.prune_method}; '
        out_str += f'merge_heuristic: {self.merge_heuristic}; '
        out_str += f'topk: {self.topk}; maxN: {self.maxN}; maxG: {self.maxG}\n'
        out_str += f'protect_pattern: {self.proctect_pattern}; '
        out_str += f'merge_cont1q: {self.merge_cont1q}; '
        out_str += f'merge_cont2q: {self.merge_cont2q}; '
        out_str += f'enable_grami: {self.enable_grami}\n\n'
        
        out_str += f'circ: {self.circ_name}; '
        out_str += f'#qubits: {self.original_circ.num_qubits}; '
        out_str += f'#gates: {self.original_circ.size()}\n'
        
        if details:
            if self.enable_grami:
                grami_result = self.result['grami']
                out_str += 'Grami:\n'
                niter = len(grami_result['grami_time'])
                for iter in range(niter):
                    out_str += f'\titer: {iter} '
                    out_str += f'time: {grami_result["grami_time"][iter]:.4f}; '
                    out_str += f'#patterns: {grami_result["nfreq_graphs"][iter]}; '
                    out_str += f'#gates: {sum(grami_result["gate_count"][iter].values())}\n'
            if self.enable_pamerge:
                if self.merge_cont1q:
                    merge_1q_result = self.result["merge1q"]
                    out_str += 'Merge 1q:\n'
                    out_str += f'time: {merge_1q_result["time"]:.4f}; '
                    out_str += f'#gates: {sum(merge_1q_result["gate_count"].values())}\n'
                if self.merge_cont2q:
                    merge_2q_result = self.result["merge2q"]
                    out_str += 'Merge 2q:\n'
                    out_str += f'time: {merge_2q_result["time"]:.4f}; '
                    out_str += f'#gates: {sum(merge_2q_result["gate_count"].values())}\n'
            
            if self.enable_pamerge:
                # change over iteratoins
                pamerge_result = self.result['pamerge']
                niter = len(pamerge_result['latency'])
                out_str += 'PAQOC:\n'
                for iter in range(niter):
                    out_str += f'\titer: {iter}:\n'
                    
                    out_str += f'\t\tgetT: {pamerge_result["get_candidates_time"][iter]:.4f}; '
                    out_str += f'selectT: {pamerge_result["select_candidates_time"][iter]:.4f}; '
                    out_str += f'mergeT: {pamerge_result["merge_candidates_time"][iter]:.4f}\n'
                    
                    out_str += f'\t\t#candidates: {len(pamerge_result["candidates"][iter])}; '
                    out_str += f'#merged: {len(pamerge_result["selected"][iter])}\n'
                    
                    
                    out_str += f'\t\tlatency: {pamerge_result["latency"][iter]:.4f}; '
                    out_str += f'compile time: {pamerge_result["compile_time"][iter]:.4f};'
                    out_str += f'#gates: {sum(pamerge_result["gate_count"][iter].values())}\n'
                
        # print final latency, compile, count
        out_str += 'Result:\n'
        out_str += f'original latency: {self.result["original_latency"]:.4f}; '
        out_str += f'new latency: {self.result["final_latency"]:.4f}\n'
        out_str += f'original #gates: {sum(self.result["original_gate_count"].values())}; '
        out_str += f'new #gates: {sum(self.result["final_gate_count"].values())}\n'
        out_str += f'compile time: {self.result["total_compile_time"]:.4f}\n'            
        
        if save_to_file is not None:
            with open(save_to_file, 'w') as f:
                f.write(out_str)
                
        print(out_str)
        
    def get_fidelity(self):
        circuit_fidelity = 1
        for node in self.qoc_dag.dag.topological_op_nodes():
            _, fidelity = self.pulse_db.get_optimal_pulse_time([node], get_fidelity=True)
            circuit_fidelity *= (1-fidelity)
        return circuit_fidelity
        
        
    
        
        
        
            

            
            
            
        
        
