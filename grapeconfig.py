from itertools import combinations
from quantum_optimal_control.core import hamiltonian

class GrapeConfig:

    def __init__(self, 
                 N, # Number of qubits
                 dt = 0.5, # unit of pulse
                 connected_qubit_pairs=None):
        self.N = N
        self.dt = dt
        if connected_qubit_pairs is None:
            self.connected_qubit_pairs = list(combinations(list(range(N)), 2))
        else:
            self.connected_qubit_pairs = connected_qubit_pairs
        self.d = 2
        self.max_iterations = 1000
        self.H0 = hamiltonian.get_H0(N, self.d)
        self.Hops, self.Hnames = hamiltonian.get_Hops_and_Hnames(N, self.d, self.connected_qubit_pairs)
        self.states_concerned_list = hamiltonian.get_full_states_concerned_list(N, self.d)
        self.maxA = hamiltonian.get_maxA(N, self.d, self.connected_qubit_pairs)
        
        self.decay =  self.max_iterations / 2
        # self.convergence = {'rate':0.01, 'update_step':10, 
        #     'max_iterations':self.max_iterations, 'conv_target':1e-3,'learning_rate_decay':self.decay} 
        # self.reg_coeffs = {'speed_up': 0.001}
        # TODO Consider use different function to specify the configuration of different systems
        self.convergence = {'rate': 0.01, 'max_iterations': self.max_iterations,
                    'conv_target':1e-3, 'learning_rate_decay':self.decay, 'min_grad': 1e-12}
        self.reg_coeffs = {'envelope': 5, 'dwdt': 0.001, 'd2wdt2': 0.00001}

              