# Create the coefficient matrices

import numpy as np
from parsing import *


class Problem:
    def __init__(self, wf: Workflows, net: LocalNetwork, cld: CloudsInfo):
        # some helpful data
        self.num_nodes = net.num + cld.num
        # The upload/download directionality is switched. For a local node sending to the cloud, that means that the
        # local node is uploading and the cloud is downloading, thereby applying cloud upload pricing, and vice, versa.
        self.prices_transfer_up = np.append(np.zeros(net.num), cld.prices_transfer_down)
        self.prices_transfer_down = np.append(np.zeros(net.num), cld.prices_transfer_up)
        self.prices_ram = np.append(np.zeros(net.num), cld.prices_ram)
        self.prices_start = np.append(np.zeros(net.num), cld.prices_start)
        self.ram_limits = np.append(np.array(net.rams), np.inf * np.ones(cld.num))
        self.p_factors = np.append(np.array(net.p_factors), cld.p_factors)

        # Coefficient matrices

        # Latency time between nodes ([second] = data[GB] / transmission rate[GB/s])
        self.L = np.zeros((wf.num_workflows, wf.num_funs, self.num_nodes, self.num_nodes))  # L_n,m,i,j
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node_sending in range(self.num_nodes):
                    for node_receiving in range(self.num_nodes):
                        transmission_rate = 0
                        if node_sending == node_receiving:
                            transmission_rate = np.inf
                        elif node_sending < net.num and node_receiving < net.num:
                            transmission_rate = net.rate_down
                        elif node_sending < net.num <= node_receiving:
                            # tinyFaaS uploading to cloud
                            transmission_rate = net.rate_up
                        elif node_sending >= net.num > node_receiving:
                            # tinyFaaS downloading from cloud
                            transmission_rate = net.rate_down
                        elif node_sending >= net.num and node_receiving >= net.num:
                            # transfer between clouds
                            transmission_rate = np.inf

                        self.L[workflow, function, node_sending, node_receiving] = \
                            wf.funs_data[workflow, function] / transmission_rate

        # Specific Time for running a function instance on a node (second)
        self.T = np.zeros((wf.num_workflows, wf.num_funs, self.num_nodes))  # T_p_n,m,i
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node in range(self.num_nodes):
                    self.T[workflow, function, node] = np.array(
                        wf.funs_times[workflow, function] / (self.p_factors[node]) * wf.funs_counts[workflow, function])

        # Cost of transferring data (cost[$] = sum{data[GB] * cost[$/GB]})
        self.D = np.zeros((wf.num_workflows, wf.num_funs, self.num_nodes, self.num_nodes))  # D_n,m,i,j
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node_sending in range(self.num_nodes):
                    for node_receiving in range(self.num_nodes):
                        if node_sending == node_receiving:
                            self.D[workflow, function, node_sending, node_receiving] = 0
                        else:
                            self.D[workflow, function, node_sending, node_receiving] = \
                                wf.funs_data[workflow, function] * self.prices_transfer_up[node_sending] \
                                + wf.funs_data[workflow, function] * self.prices_transfer_down[node_receiving]

        # Cost of running a function ($ = price_ram[$/GB] * (fun_size[GB] + fun_data[GB]))
        self.C = np.zeros((wf.num_workflows, wf.num_funs, self.num_nodes))  # C_n,m,i
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node in range(self.num_nodes):
                    self.C[workflow, function, node] = self.prices_ram[node] * self.T[workflow, function][node] * \
                                                       (wf.funs_sizes[workflow, function]
                                                        + wf.funs_data[workflow, function])

        # Cost of starting a function ($/start)
        self.C_s = np.zeros((wf.num_workflows, wf.num_funs, self.num_nodes))  # C_s_n,m,i
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node in range(self.num_nodes):
                    self.C_s[workflow, function, node] = self.prices_start[node]
