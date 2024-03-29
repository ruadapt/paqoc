import numpy as np
import tensorflow as tf
from .analysis import Analysis
import os
import time
from scipy.optimize import minimize

from quantum_optimal_control.helper_functions.data_management import H5File


class run_session:
    def __init__(self, tfs, graph, conv, sys_para, method, show_plots=True, single_simulation=False, use_gpu=True):
        self.tfs = tfs
        self.graph = graph
        self.conv = conv
        self.sys_para = sys_para
        self.update_step = conv.update_step
        self.iterations = 0
        self.method = method.upper()
        self.show_plots = show_plots
        self.target = False

        if not use_gpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = None
        
        with tf.Session(graph=graph, config=config) as self.session:

            tf.global_variables_initializer().run()

            # print("Initialized", flush=True)

            if self.method == 'EVOLVE':
                self.start_time = time.time()
                x0 = self.sys_para.ops_weight_base
                self.l, self.rl, self.grads, self.metric, self.g_squared = self.get_error(
                    x0)
                self.get_end_results()

            else:
                if self.method != 'ADAM':  # Any BFGS scheme
                    self.bfgs_optimize(method=self.method)

                if self.method == 'ADAM':
                    self.start_adam_optimizer()

    def start_adam_optimizer(self):
        # adam optimizer
        self.start_time = time.time()
        self.end = False
        while True:

            self.g_squared, self.l, self.rl, self.metric = self.session.run(
                [self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale])

            if (self.l < self.conv.conv_target) or (self.g_squared < self.conv.min_grad) \
                    or (self.iterations >= self.conv.max_iterations):
                self.end = True

            self.update_and_save()

            if self.end:
                self.get_end_results()
                break

            learning_rate = float(
                self.conv.rate) * np.exp(-float(self.iterations) / self.conv.learning_rate_decay)
            self.feed_dict = {self.tfs.learning_rate: learning_rate}

            _ = self.session.run([self.tfs.optimizer],
                                 feed_dict=self.feed_dict)

    def update_and_save(self):

        if not self.end:

            if (self.iterations % self.conv.update_step == 0):
                self.anly = Analysis(self.sys_para, self.tfs.final_state, self.tfs.ops_weight, self.tfs.unitary_scale,
                                     self.tfs.inter_vecs)
                self.save_data()
                self.display()
            if (self.iterations % self.conv.evol_save_step == 0):
                if not (self.sys_para.show_plots == True and (self.iterations % self.conv.update_step == 0)):
                    self.anly = Analysis(self.sys_para, self.tfs.final_state, self.tfs.ops_weight, self.tfs.unitary_scale,
                                         self.tfs.inter_vecs)
                    if not (self.iterations % self.conv.update_step == 0):
                        self.save_data()
                    self.conv.save_evol(self.anly)

            self.iterations += 1

    def get_end_results(self):
        # get optimized pulse and propagation

        # get and save inter vects

        self.anly = Analysis(self.sys_para, self.tfs.final_state, self.tfs.ops_weight, self.tfs.unitary_scale,
                             self.tfs.inter_vecs)
        self.save_data()
        self.display()
        if not self.show_plots:
            self.conv.save_evol(self.anly)

        self.uks = self.Get_uks()
        if not self.sys_para.state_transfer:
            self.Uf = self.anly.get_final_state()
        else:
            self.Uf = []

    def Get_uks(self):
        # to get the pulse amplitudes
        uks = self.anly.get_ops_weight()
        for ii in range(len(uks)):
            uks[ii] = self.sys_para.ops_max_amp[ii]*uks[ii]
        return uks

    def get_error(self, uks):
        # get error and gradient for scipy bfgs:
        self.session.run(self.tfs.ops_weight_base.assign(uks))

        g, l, rl, metric, g_squared = self.session.run(
            [self.tfs.grad_pack, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale, self.tfs.grad_squared])

        final_g = np.transpose(np.reshape(
            g, (len(self.sys_para.ops_c)*self.sys_para.steps)))

        return l, rl, final_g, metric, g_squared

    def save_data(self):
        if self.sys_para.save:
            self.elapsed = time.time() - self.start_time
            with H5File(self.sys_para.file_path) as hf:
                hf.append('error', np.array(self.l))
                hf.append('reg_error', np.array(self.rl))
                hf.append('uks', np.array(self.Get_uks()))
                hf.append('iteration', np.array(self.iterations))
                hf.append('run_time', np.array(self.elapsed))
                hf.append('unitary_scale', np.array(self.metric))

    def display(self):
        # display of simulation results

        if self.show_plots:
            self.conv.update_plot_summary(self.l, self.rl, self.anly)
        else:
            print('Error = :%1.2e; Runtime: %.1fs; Iterations = %d, grads =  %10.3e, unitary_metric = %.5f' % (self.l, self.elapsed, self.iterations, self.g_squared, self.metric), flush=True)

    def minimize_opt_fun(self, x):
        # minimization function called by scipy in each iteration
        self.l, self.rl, self.grads, self.metric, self.g_squared = self.get_error(
            np.reshape(x, (len(self.sys_para.ops_c), int(len(x)/len(self.sys_para.ops_c)))))

        if self.l < self.conv.conv_target:
            self.conv_time = time.time()-self.start_time
            self.conv_iter = self.iterations
            self.end = True
            print('Target fidelity reached', flush=True)
            self.grads = 0*self.grads  # set zero grads to terminate the scipy optimization

        self.update_and_save()

        if self.method == 'L-BFGS-B':
            return np.float64(self.rl), np.float64(np.transpose(self.grads))
        else:
            return self.rl, np.reshape(np.transpose(self.grads), [len(np.transpose(self.grads))])

    def bfgs_optimize(self, method='L-BFGS-B', jac=True, options=None):
        # scipy optimizer
        self.conv.reset_convergence()
        self.first = True
        self.conv_time = 0.
        self.conv_iter = 0
        self.end = False
        print("Starting " + self.method + " Optimization", flush=True)
        self.start_time = time.time()

        x0 = self.sys_para.ops_weight_base
        options = {'maxfun': self.conv.max_iterations,
                   'gtol': self.conv.min_grad, 'disp': False, 'maxls': 40}

        res = minimize(self.minimize_opt_fun, x0,
                       method=method, jac=jac, options=options)

        uks = np.reshape(res['x'], (len(self.sys_para.ops_c),
                                    int(len(res['x'])/len(self.sys_para.ops_c))))

        print(self.method + ' optimization done', flush=True)

        g, l, rl = self.session.run(
            [self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss])

        if self.sys_para.show_plots == False:
            print(res.message, flush=True)
            print(("Error = %1.2e" % l), flush=True)
            print(("Total time is " + str(time.time() - self.start_time)), flush=True)

        self.get_end_results()


# _tf_cache_file_name = "tf_cache_.cpkt"
# def _get_tfgraph(tfs, cache_graph, data_path):
#     """Return the appropriate tensorflow graph. If cache is true
#     and a cache exists, that cache will be used to get an existing tf graph.
#     If cache is true and no cache exists, a new graph will be created and
#     cached. If cache is false a new graph will be generated.
#     Args:
#     tfs :: quantum_optimal_control.core.tensorflow_state.TensorflowState
#         - the tensorflow state of the optimization
#     cache_graph :: bool - wether or not to attempt lookup of a cached graph
#     data_path :: str - the path to the cache (assumes the path exists)

#     Returns: graph :: ? - the tensorflow state and its corresponding graph
#     """
#     with tf.device(self.sys_para.device)
#     graph = tfs.build_graph()
#     tf_cache_file_path = os.path.join(data_path, _graph_cache_file_name)
#     if cache_graph:
#         # If the file exists, start the session from the 
#         if os.path.isfile(graph_cache_file):
#             tfs.saver.save(sess, grapc)
#             print("Using cached tensorflow graph at: {}"
#                   "".format(graph_cache_file))
#             with open(graph_cache_file, 'rb') as f:
#                 graph = pickle.load(f)
#         # If the file does not exist, generate a new graph and cache it.
#         else:
#             print("Caching tensorflow graph at: {}"
#                   "".format(graph_cache_file))
#             graph = tfs.build_graph()
#             data = tfs, graph
#             with open(graph_cache_file, 'wb') as f:
#                 pickle.dump(data, f, protocol=2, fix_imports=True)
            
#     # If the graph should not be cached, generate a new one.
#     else:
#         print("Generating new tensorflow graph.")
#         graph = tfs.build_graph()

#     return graph
            

