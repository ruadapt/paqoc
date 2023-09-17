from typing import Dict, List, Tuple

from qiskit.dagcircuit.dagnode import DAGNode
from qiskit.quantum_info.operators import Operator

from quantum_optimal_control.main_grape.grape import Grape
import pandas as pd
import numpy as np

import time
import hashlib

from helper import nodes2circ, c_to_r_mat, unitary2circ_key
from grapeconfig import GrapeConfig

class PulseDB:

    def __init__(self, csv_file: str) -> None:
        self.csv_file = csv_file
        self.pulse_db = pd.read_csv(self.csv_file)
        self.pulse_runtime_dict: Dict[str, float] = dict()
        self.prestored_compile_time = 0

    def __binary_search_for_shortest_pulse_time(self, U, grape_config, initial_guess=[]):
        """Search between [min_steps, max_steps] (inclusive)."""
        steps = [0, 0, 20, 40, 60, 120, 200]
        nn = grape_config.N

        checked = set()
        all_records = []
        valid_records = []
        latency = None
        while nn not in checked:
            checked.add(nn)
            if nn + 1 >= len(steps) or nn < 0:
                break

            min_steps = steps[nn]
            max_steps = steps[nn + 1]
            delta = max(int((max_steps - min_steps)/20), 1)

            iter = 0
            res_ = None
            
            while min_steps + delta < max_steps:  # just estimate to +- dt * 2 ns
                iter += 1
                mid_steps = int((min_steps + max_steps) / 2)
                total_time = mid_steps * grape_config.dt
                
                print(f'iter: {iter}, steps: {mid_steps} / [{min_steps}-{max_steps}] total_time: {total_time}')
                # logging.info('iter: %d, steps: %d / [%d-%d] total_time: %.4f',
                #              iter, mid_steps, min_steps, max_steps, total_time)

                res = Grape(grape_config.H0, grape_config.Hops, grape_config.Hnames, U, 
                            total_time, mid_steps, 
                            grape_config.states_concerned_list, grape_config.convergence,
                            reg_coeffs=grape_config.reg_coeffs, initial_guess = initial_guess,
                            use_gpu=False, sparse_H=False, method='ADAM', maxA=grape_config.maxA,
                            save=False, show_plots=False, return_converged=True)

                print(total_time, res.l)
                all_records.append((res.l, total_time, res))
                if res.l <= grape_config.convergence['conv_target']:
                    max_steps = mid_steps
                    valid_records.append((total_time, res))
                else: 
                    min_steps = mid_steps

            if len(valid_records) == 0:
                nn = nn + 1
            elif nn != 0 and min_steps == steps[nn]:
                nn = nn - 1
            else:
                break
        # FIXME this is a walkaround
        if len(valid_records) == 0:
            try:
                min_result = min(all_records)
                return min_result[1], min_result[2]
            except:
                raise Exception('Grape fails to get the pulse.')
        else:
            min_result = min(valid_records)
            return min_result[0], min_result[1]
    
    
    def read_pulse_from_db(self, circ_key_hash, N=None):
        '''Return the pulse info if the record is found in pulse_db. Otherwise return None'''
        if N is None:
            pulse_candidates = self.pulse_db[self.pulse_db['circ_key_hash'] == circ_key_hash]
        else:
            pulse_candidates = self.pulse_db[(self.pulse_db['N'] == N) & (self.pulse_db['circ_key_hash'] == circ_key_hash)]
        
        if pulse_candidates.empty:
            return None
        else:
            pulse_data_lst = pulse_candidates.to_dict(orient='records')
            pulse_data = min(pulse_data_lst, key = lambda x: x['latency'])

            return pulse_data

    def write_pulse_to_db(self, U, circ_key, circ_key_hash, circ, grape_config, fidelity, latency, compile_time):
        pulse_data = {'circ_key': circ_key, 'circ_qasm': circ.qasm(), 
                        'N' : grape_config.N, 'dt' : grape_config.dt, 
                        # 'initial_guess': grape_config.initial_guess, 
                        'fidelity': fidelity, 
                        'latency' : latency, 
                        'compile_time' : compile_time,
                        'circ_key_hash': circ_key_hash}
        self.pulse_db = self.pulse_db.append(pulse_data, ignore_index=True)  

    def save_pulsedb(self):
        self.pulse_db.to_csv(self.csv_file, mode='w', header=True, index=False)

    def get_prestored_compile_time(self) -> float:
        return self.prestored_compile_time
    
    def reset_prestored_compile_time(self):
        self.prestored_compile_time = 0

    def get_optimal_pulse_time(self, 
                               nodes: List[DAGNode], 
                               initial_guess = [],
                               approx: bool = False,
                               get_fidelity = False) -> Tuple[float, float]:
        '''
        Return the optimal controlled pulse for the given node
        [Latency, CompileTime]
        '''

        # Convert DAGNode to a unitray matrix
        _, circ, _, new_name = nodes2circ(nodes) 
        N = circ.num_qubits
        if approx: return N, 0
        # logging.info(N)
        # logging.info(circ.draw())
                
        U = Operator(circ).data
        # Genearte GrapeConfig

        # FIXME Move this to __init__
        dt = 0.5
        grape_config = GrapeConfig(N, dt)
        
        circ_key, circ_key_hash = unitary2circ_key(U)
        
        if circ_key_hash in self.pulse_runtime_dict and get_fidelity is False:
            return self.pulse_runtime_dict[circ_key_hash], 0
        
        # FIXME: add initial guess
        if len(initial_guess) == 0:
            pass
        
        pulse_data = self.read_pulse_from_db(circ_key_hash, N)
        fidelity = 1
        if pulse_data is None:
            # logging.info('circ_key: %s/%s/N:%d is not found in pulse_db, calculate the pulse.',
            #              circ_key, circ_key_hash, N)
            # print(f'circ_key:{circ_key}/{circ_key_hash}/N:{N} is not found in pulse_db, calculate the pulse.')
            # print(f'Get node {new_name} pulse...', end='')
            start = time.time()
            latency, res = self.__binary_search_for_shortest_pulse_time(U, grape_config)
            # print(latency)
            stop = time.time()
            compile_time = stop - start

            # update pulse_db
            if latency is not None and res is not None:
                try:
                    self.write_pulse_to_db(U, circ_key, circ_key_hash, circ, grape_config, (res.__dict__)['l'], latency, compile_time)
                    self.save_pulsedb()
                except:
                    pass
                    # logging.error('Faill to write to pulsedb. latency %d name %s.', latency, new_name)
            elif latency is None:
                latency, compile_time = 10 * N, 0
        else:
            # print(f'circ_key:{circ_key}/{circ_key_hash}/N:{N} found in pulse_db.')
            latency, compile_time, fidelity = pulse_data['latency'], pulse_data['compile_time'], pulse_data['fidelity']
            self.prestored_compile_time += compile_time
        
        self.pulse_runtime_dict[circ_key_hash] = latency
        
        return latency, fidelity