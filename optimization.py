from gurobipy import *
from creation import *


class Optimizer:
    def __init__(self, wf: Workflows, net: LocalNetwork, pb: Problem):
        self.model = Model("TinyFaaS-Schedule")
        self.P = self.model.addVars(wf.num_workflows, wf.num_funs, pb.num_nodes,
                                    vtype=GRB.INTEGER, name="P")  # P_n,m,i

        # expressions for L, and data transfer costs
        obj_latency = 0
        obj_transfer = 0
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs - 1):
                for node_sending in range(pb.num_nodes):
                    for node_receiving in range(pb.num_nodes):
                        obj_latency = obj_latency + self.P[workflow, function, node_sending] \
                                      * self.P[workflow, function + 1, node_receiving] \
                                      * pb.L[node_sending, node_receiving]
                        obj_transfer = obj_transfer + self.P[workflow, function, node_sending]\
                                       * self.P[workflow, function + 1, node_receiving]\
                                       * pb.D[workflow, function, node_sending, node_receiving]

        # expressions for time, and data running costs
        obj_time = 0
        obj_ram = 0
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node in range(pb.num_nodes):
                    obj_time = obj_time + self.P[workflow, function, node] * pb.T[workflow, function, node]
                    obj_ram = obj_ram + self.P[workflow, function, node] * pb.C[workflow, function, node]

        # expression for the total objective function
        self.w_1, self.w_2 = 1, 470000  # w1 for L and time, w2 for costs
        obj = self.w_1 * (obj_latency + obj_time) + self.w_2 * (obj_transfer + obj_ram)
        self.model.setObjective(obj, GRB.MINIMIZE)

        # Constraints

        # start and end functions should be assigned to TinyFaaS nodes
        # P_n,(0,5,7),n = 1 for all n else 0
        for workflow in range(wf.num_workflows):
            for function in wf.funs_local[workflow]:
                self.model.addConstr(self.P[workflow, function, workflow] - 1 == 0)

        # training functions can not be executed in local nodes
        # sum_k P_n,4,k = 0 , k in {0 .. 3}
        for workflow in range(wf.num_workflows):
            for function in wf.funs_cloud[workflow]:
                self.model.addConstr(quicksum(self.P[workflow, function, node] for node in range(net.num_tiny)) == 0)

        # maximum ram limit per sum of functions m on node k
        # sum_n sum_k P_n,m,k * RAM_n,m <= MAX_k , for all k in {0 ..3} for all n, m
        for node in range(pb.num_nodes):
            self.model.addConstr(
                quicksum(quicksum(self.P[workflow, function, node] * wf.funs_data[workflow][function]
                                  for function in range(wf.num_funs))
                         for workflow in range(wf.num_workflows)) <= pb.ram_limits[node])

        # each function can only be assigned to a single node
        # sum_j P_n,m,j = 1 for all n,m
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                self.model.addConstr(quicksum(self.P[workflow, function, node] for node in range(pb.num_nodes))
                                     == wf.funs_counts[workflow][function])


    # Solve model
    def solve(self):
        self.model.setParam("OutputFlag", 1)
        self.model.update()
        self.model.optimize()
