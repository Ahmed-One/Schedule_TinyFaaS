# Create the coefficient matrices

import numpy as np
from parsing import *


class Problem:
    def __init__(self, wf: Workflows, net: LocalNetwork, cld: CloudsInfo):
        # some helpful data
        self.num_nodes = net.num_tiny + cld.num_clouds
        self.prices_transfer = np.append(np.zeros(net.num_tiny), cld.prices_cloud_transfer)
        self.prices_ram = np.append(np.zeros(net.num_tiny), cld.prices_cloud_ram)
        self.prices_start = np.append(np.zeros(net.num_tiny), cld.prices_cloud_start)
        self.ram_limits = np.append(np.array(net.rams_tiny), np.inf * np.ones(cld.num_clouds))

        # Coefficient matrices
        # Latency between nodes ([second] = tiny-to-router[second] + router-to-cloud[second])
        self.L = np.zeros((self.num_nodes, self.num_nodes))  # L_i,j
        latency_list = np.append(net.latency_local * np.ones(net.num_tiny), np.array(cld.latency_cloud))
        for node_sending in range(self.num_nodes):
            for node_receiving in range(self.num_nodes):
                if node_sending == node_receiving:
                    self.L[node_sending, node_receiving] = 0
                elif node_sending < net.num_tiny and node_receiving < net.num_tiny:
                    self.L[node_sending, node_receiving] = net.latency_local
                else:
                    self.L[node_sending, node_receiving] = latency_list[node_sending] + latency_list[node_receiving]

        # Time for running a function on a node (second)
        self.T = np.zeros((wf.num_workflows, wf.num_funs, self.num_nodes))  # T_n,m,j
        for workflow in range(wf.num_workflows):
            for node in range(self.num_nodes):
                self.T[workflow, :, node] = np.array(wf.funs_times[workflow])

        # Cost of transferring data (cost[$] = sum{data[GB] * cost[$/GB]})
        self.D = np.zeros((wf.num_workflows, wf.num_funs, self.num_nodes, self.num_nodes))  # D_n,m,j,i
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node_sending in range(self.num_nodes):
                    for node_receiving in range(self.num_nodes):
                        if node_sending == node_receiving:
                            self.D[workflow, function, node_sending, node_receiving] = 0
                        else:
                            self.D[workflow, function, node_sending, node_receiving] = \
                                    wf.funs_data[workflow][function] * self.prices_ram[node_sending] + \
                                    wf.funs_data[workflow][function] * self.prices_ram[node_receiving]

        # Cost of running a function ($ = price_start[$/num] * fun_count + fun_size[GB] * price_ram[$/GB] * fun_time[s])
        self.C = np.zeros((wf.num_workflows, wf.num_funs, self.num_nodes))  # C_n,m,j
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node in range(self.num_nodes):
                    self.C[workflow, function, node] = self.prices_start[node] * wf.funs_counts[workflow][function]\
                                                       + wf.funs_sizes[workflow][function] * self.prices_ram[node]\
                                                       * wf.funs_times[workflow][function]
