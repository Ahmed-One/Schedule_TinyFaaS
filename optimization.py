from gurobipy import *
from creation import *


class Optimization:
    def __init__(self):
        self.model = Model("TinyFaaS-Schedule")

        # Objective costs
        self.obj = 0
        self.obj_latency = 0
        self.obj_transfer = 0
        self.obj_latency_workflows = []
        self.obj_transfer_workflows = []
        self.obj_time = 0
        self.obj_ram = 0
        self.obj_time_workflows = []
        self.obj_ram_workflows = []

        # Objective function weights
        # w1 for L and time, w2 for costs
        self.w_1, self.w_2 = 1, 470000

    def formulate_objectives(self):
        # expression for the total objective function
        self.obj = self.w_1 * (self.obj_latency + self.obj_time) + self.w_2 * (self.obj_transfer + self.obj_ram)
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    # Solve model
    def solve(self):
        self.model.setParam("OutputFlag", 1)
        self.model.update()
        self.model.optimize()


class Optimizer(Optimization):
    def __init__(self, wf: Workflows, net: LocalNetwork, pb: Problem):
        super().__init__()
        self.P = self.model.addVars(wf.num_workflows, wf.num_funs, pb.num_nodes,
                                    vtype=GRB.INTEGER, name="P")  # P_n,m,i

        # expressions for L, and data transfer costs
        for workflow in range(wf.num_workflows):
            obj_latency_functions = []
            obj_transfer_functions = []
            for function in range(wf.num_funs - 1):
                obj_latency_nodes = []
                obj_transfer_nodes = []
                for node_sending in range(pb.num_nodes):
                    obj_latency = 0
                    obj_transfer = 0
                    for node_receiving in range(pb.num_nodes):
                        obj_latency = obj_latency + self.P[workflow, function, node_sending] \
                                      * self.P[workflow, function + 1, node_receiving] \
                                      * pb.L[node_sending, node_receiving]
                        self.obj_latency += obj_latency

                        obj_transfer = obj_transfer + self.P[workflow, function, node_sending] \
                                       * self.P[workflow, function + 1, node_receiving] \
                                       * pb.D[workflow, function, node_sending, node_receiving]
                        self.obj_transfer += obj_transfer

                    obj_latency_nodes.append(obj_latency)
                    obj_transfer_nodes.append(obj_transfer)
                obj_latency_functions.append(obj_latency_nodes)
                obj_transfer_functions.append(obj_transfer_nodes)
            self.obj_latency_workflows.append(obj_latency_functions)
            self.obj_transfer_workflows.append(obj_transfer_functions)

        # expressions for time, and data running costs
        for workflow in range(wf.num_workflows):
            obj_time_functions = []
            obj_ram_functions = []
            for function in range(wf.num_funs):
                obj_time_nodes = []
                obj_ram_nodes = []
                for node in range(pb.num_nodes):
                    obj_time = self.P[workflow, function, node] * pb.T[workflow, function, node]
                    self.obj_time += obj_time

                    obj_ram = self.P[workflow, function, node] * pb.C[workflow, function, node]
                    self.obj_ram += obj_ram

                    obj_time_nodes.append(obj_time)
                    obj_ram_nodes.append(obj_ram)
                obj_time_functions.append(obj_time_nodes)
                obj_ram_functions.append(obj_ram_nodes)
            self.obj_time_workflows.append(obj_time_functions)
            self.obj_ram_workflows.append(obj_ram_functions)

        self.formulate_objectives()

        # Constraints

        # start and end functions should be assigned to TinyFaaS nodes
        # P_n,(0,5,7),n = 1 for all n else 0
        for workflow in range(wf.num_workflows):
            for function in wf.funs_local[workflow]:
                self.model.addConstr(self.P[workflow, function, workflow] - 1 == 0,
                                     name='local-function-constraint')

        # training functions can not be executed in local nodes
        # sum_k P_n,4,k = 0 , k in {0 .. 3}
        for workflow in range(wf.num_workflows):
            for function in wf.funs_cloud[workflow]:
                self.model.addConstr(quicksum(self.P[workflow, function, node] for node in range(net.num_tiny)) == 0,
                                     name='cloud-function-constraint')

        # maximum ram limit per sum of functions m on node k
        # sum_n sum_m P_n,m,k * RAM_n,m <= MAX_k , for all k in {0 ..3} for all n, m
        for node in range(pb.num_nodes):
            self.model.addConstr(
                quicksum(quicksum(self.P[workflow, function, node] * wf.funs_data[workflow][function]
                                  for function in range(wf.num_funs))
                         for workflow in range(wf.num_workflows)) <= pb.ram_limits[node],
                name='ram-limit-constraint')

        # each function can only be assigned to a single node
        # sum_j P_n,m,j = 1 for all n,m
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                self.model.addConstr(quicksum(self.P[workflow, function, node] for node in range(pb.num_nodes))
                                     == wf.funs_counts[workflow][function],
                                     name='function-locality-constraint')


# Optimizer2 optimizes deployment for edges not nodes to eliminate non-linear terms
class Optimizer2(Optimization):
    def __init__(self, wf: Workflows, net: LocalNetwork, pb: Problem):
        super().__init__()
        self.P = self.model.addVars(wf.num_workflows, wf.num_funs, pb.num_nodes, pb.num_nodes,
                                    vtype=GRB.INTEGER, name="P")  # P_n,m,i,j

        # expressions for L, and data transfer costs
        for workflow in range(wf.num_workflows):
            obj_latency_functions = []
            obj_transfer_functions = []
            for function in range(wf.num_funs):
                obj_latency_nodes = []
                obj_transfer_nodes = []
                for node_sending in range(pb.num_nodes):
                    obj_latency = 0
                    obj_transfer = 0
                    for node_receiving in range(pb.num_nodes):
                        obj_latency = obj_latency + self.P[workflow, function, node_sending, node_receiving] \
                                      * pb.L[node_sending, node_receiving]
                        self.obj_latency += obj_latency

                        obj_transfer = obj_transfer + self.P[workflow, function, node_sending, node_receiving] \
                                       * pb.D[workflow, function, node_sending, node_receiving]
                        self.obj_transfer += obj_transfer

                    obj_latency_nodes.append(obj_latency)
                    obj_transfer_nodes.append(obj_transfer)
                obj_latency_functions.append(obj_latency_nodes)
                obj_transfer_functions.append(obj_transfer_nodes)
            self.obj_latency_workflows.append(obj_latency_functions)
            self.obj_transfer_workflows.append(obj_transfer_functions)

        # expressions for time, and data running costs
        for workflow in range(wf.num_workflows):
            obj_time_functions = []
            obj_ram_functions = []
            for function in range(wf.num_funs):
                obj_time_nodes = []
                obj_ram_nodes = []
                for node_sending in range(pb.num_nodes):
                    for node_receiving in range(pb.num_nodes):
                        obj_time = self.P[workflow, function, node_sending, node_receiving] \
                                   * pb.T[workflow, function, node_sending]
                        self.obj_time += obj_time

                        obj_ram = self.P[workflow, function, node_sending, node_receiving] \
                                  * pb.C[workflow, function, node_sending]
                        self.obj_ram += obj_ram

                        obj_time_nodes.append(obj_time)
                        obj_ram_nodes.append(obj_ram)
                obj_time_functions.append(obj_time_nodes)
                obj_ram_functions.append(obj_ram_nodes)
            self.obj_time_workflows.append(obj_time_functions)
            self.obj_ram_workflows.append(obj_ram_functions)

        # expression for the total objective function
        self.formulate_objectives()

        # Constraints

        # start and end functions should be assigned to TinyFaaS nodes
        # P_n,(0,5,7),n,j = 1 for all n else 0
        for workflow in range(wf.num_workflows):
            for function in wf.funs_local[workflow]:
                self.model.addConstr(quicksum(self.P[workflow, function, workflow, node_receiving]
                                              for node_receiving in range(pb.num_nodes)) - 1 == 0,
                                     name='local-function-constraint')

        # training functions can not be executed in local nodes
        # sum_k P_n,4,k,j = 0 , k in {0 .. 3} and all j
        for workflow in range(wf.num_workflows):
            for function in wf.funs_cloud[workflow]:
                self.model.addConstr(quicksum(quicksum(self.P[workflow, function, node_sending, node_receiving]
                                                       for node_sending in range(net.num_tiny))
                                              for node_receiving in range(pb.num_nodes)) == 0,
                                     name='cloud-function-constraint')

        # maximum ram limit per sum of functions m on node k
        # sum_n sum_m sum_j P_n,m,i,j * RAM_n,m <= RAM_MAX_i , for all i
        for node_sending in range(pb.num_nodes):
            self.model.addConstr(quicksum(quicksum(quicksum(
                self.P[workflow, function, node_sending, node_receiving] * wf.funs_data[workflow][function]
                for node_receiving in range(pb.num_nodes))
                                                   for function in range(wf.num_funs))
                                          for workflow in range(wf.num_workflows)) <= pb.ram_limits[node_sending],
                                 name='ram-limit-constraint')

        # each function can only be assigned to a single node
        # sum_j P_n,m,i,j = 1 for all n,m
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                self.model.addConstr(quicksum(quicksum(self.P[workflow, function, node_sending, node_receiving]
                                                       for node_receiving in range(pb.num_nodes))
                                              for node_sending in range(pb.num_nodes))
                                     == wf.funs_counts[workflow][function],
                                     name='function-locality-constraint')

        # Continuation Constraint?
        # sum_i P_n,m,i,j = sum_k P_n,m+1,j,k
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs - 1):
                for node_receiving in range(pb.num_nodes):
                    self.model.addConstr(quicksum(self.P[workflow, function, node_sending, node_receiving]
                                                  for node_sending in range(pb.num_nodes)) ==
                                         quicksum(self.P[workflow, function + 1, node_receiving, node_receiving_next]
                                                  for node_receiving_next in range(pb.num_nodes)),
                                         name='edge-consistency-constraint?')
