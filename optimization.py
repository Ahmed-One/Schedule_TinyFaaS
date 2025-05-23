from abc import abstractmethod

import numpy as np
from gurobipy import *

from creation import *


class Optimization:
    def __init__(self, var_dim: tuple):
        self.model = Model("TinyFaaS-Schedule")
        self.report = {}
        self.var_dim = var_dim
        self.type = GRB.INTEGER
        self.P = self.model.addVars(*self.var_dim, vtype=self.type, name="P")

        # Optimizer normally does not optimize for batch processing
        self.flagBatch = False

        # Objective costs
        self.obj: LinExpr = 0 * LinExpr()  # Total objective cost
        self.obj_latency: LinExpr = 0 * LinExpr()  # Time after function run ends before other function can start
        self.obj_transfer: LinExpr = 0 * LinExpr()  # Cost($) of transferring data between nodes
        self.obj_latency_details: {LinExpr} = {}  # Latency objective costs categorized
        self.obj_transfer_details: {LinExpr} = {}  # Data transfer costs categorized
        self.obj_time: LinExpr = 0 * LinExpr()  # Workflow-function run time
        self.obj_ram: LinExpr = 0 * LinExpr()  # Function running cost($)
        self.obj_time_details: {LinExpr} = {}  # Function run time categorized
        self.obj_ram_details: {LinExpr} = {}  # Function running cost categorized

        self.obj_startup_time: LinExpr = 0 * LinExpr()  # objective for the startup time
        self.obj_startup_cost: LinExpr = 0 * LinExpr()  # objective for the startup cost
        self.obj_startup_cost_details: {LinExpr} = {}  # Startup cost categorized
        self.obj_control: LinExpr = 0 * LinExpr()  # Objective monetary cost of online-control
        self.obj_control_details: {LinExpr} = {}  # Online-control cost categorized

        # Objective function weights
        # w1 for latency and time, w2 for costs
        self.w_1, self.w_2 = 1, 470000

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    # Sum up formulated objectives
    def sum_objectives(self):
        # expression for the total objective function
        self.obj = self.w_1 * (self.obj_latency + self.obj_time + self.obj_startup_time)\
                   + self.w_2 * (self.obj_transfer + self.obj_ram + self.obj_startup_cost + self.obj_control)
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def normalize_weights(self, wf: Workflows, pb: Problem):
        # Normalize by finding a utopia  and point and applying a norm
        # (Grodzevich and Romanko: Normalization and other topics)

        # Find utopia and nadir points for time objectives
        temp_obj_times = self.obj_latency + self.obj_time
        self.model.setObjective(temp_obj_times, GRB.MINIMIZE)
        self.model.update()
        self.model.optimize()
        utopia_time = temp_obj_times.getValue()

        # Find utopia and nadir points for cost objectives
        temp_obj_costs = self.obj_transfer + self.obj_ram
        self.model.setObjective(temp_obj_costs, GRB.MINIMIZE)
        self.model.update()
        self.model.optimize()
        utopia_cost = temp_obj_costs.getValue()

    def normalize_weights_simple(self, wf: Workflows, pb: Problem):
        sum_time = 0
        sum_cost = 0
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                sum_time += wf.funs_counts[workflow, function] \
                            * (np.mean(pb.T[workflow, function]) + np.mean(pb.L[workflow, function]))
                sum_cost += wf.funs_counts[workflow, function] \
                            * (np.mean(pb.D[workflow, function]) + np.mean(pb.C[workflow, function]))

        ratio_timecost = sum_time / sum_cost
        self.w_2 = ratio_timecost
        self.w_1 = 1

    # Solve model
    def solve(self):
        self.sum_objectives()
        # self.model.params.NumericFocus = 3
        self.model.setParam("OutputFlag", 1)
        self.model.update()

        # Get presolve statistics
        self.report['init_vars'] = self.model.NumVars
        self.report['init_const'] = self.model.NumConstrs
        presolved_model = self.model.presolve()
        self.report['presolve_vars'] = presolved_model.NumVars
        self.report['presolve_constr'] = presolved_model.NumConstrs

        self.model.optimize()

        # Get solve statistics
        self.report['num_vars'] = self.model.NumVars
        self.report['num_constr'] = self.model.NumConstrs
        self.report['num_GenConstr'] = self.model.NumGenConstrs
        self.report['runtime'] = self.model.Runtime
        self.report['simplex_iterations'] = self.model.IterCount
        self.report['BranchnBound_nodes'] = self.model.NodeCount
        self.report['Optimality_Gap'] = self.model.MIPGap

    # Constraint formulation methods

    def constrain_to_local_nodes(self, wf: Workflows, net: LocalNetwork):
        """
        Local functions should be assigned to local TinyFaaS nodes
        sum_i P_n,local,i = 1 for all i in {local-nodes}
        :param wf: (Workflows object) workflows data
        :param net: (LocalNetwork object) local network nodes data
        :return: None
        """
        for workflow in range(wf.num_workflows):
            for function in wf.funs_local[workflow]:
                self.model.addConstr(quicksum(self.P[workflow, function, node] for node in range(net.num))
                                     == wf.funs_counts[workflow, function],
                                     name='local-function-constraint')

    def constrain_to_cloud_nodes(self, wf: Workflows, net: LocalNetwork):
        """
        Training functions can not be executed in local nodes
        sum_k P_n,cloud,k = 0 , k in {local-nodes}
        :param wf: (Workflows object) workflows data
        :param net: (LocalNetwork object) local network nodes data
        :return: None
        """
        for workflow in range(wf.num_workflows):
            for function in wf.funs_cloud[workflow]:
                self.model.addConstr(quicksum(self.P[workflow, function, node] for node in range(net.num)) == 0,
                                     name='cloud-function-constraint')

    def constrain_to_function_count(self, wf: Workflows, pb: Problem):
        """
        Each function can only be assigned as much as allowable instance count
        sum_j P_n,m,j = count_n,m for all n,m
        :param wf: (Workflows object) workflows data
        :param pb: (Problem object) combined problem data
        :return: None
        """
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                self.model.addConstr(quicksum(self.P[workflow, function, node] for node in range(pb.num_nodes))
                                     == wf.funs_counts[workflow, function],
                                     name='function-locality-constraint')

    def constrain_to_ram_limit(self, wf: Workflows, pb: Problem):
        """
        maximum ram limit per sum of functions m on node k
        sum_n sum_m P_n,m,k * RAM_n,m <= MAX_k , for all k
        :param wf: (Workflows object) workflows data
        :param pb: (Problem object) combined problem data
        :return: None
        """
        for node in range(pb.num_nodes):
            self.model.addConstr(
                quicksum(quicksum(self.P[workflow, function, node] * (wf.funs_data[workflow, function]
                                                                      + wf.funs_sizes[workflow, function])
                                  for function in range(wf.num_funs))
                         for workflow in range(wf.num_workflows)) <= pb.ram_limits[node],
                name='ram-limit-constraint')

    # Get data for visualization
    def get_result(self, v: str = 'P'):
        if v == 'P':
            dims = self.var_dim
        elif v in {'x', 'y'}:
            # workflows x function x nodes
            dims = (self.var_dim[0], 1, self.var_dim[-1])

        vs = [var for var in self.model.getVars() if v in var.VarName and var.VType == self.type]
        return np.reshape(np.array([item.x for item in vs]), dims)


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# Initial formulation. P is binary. See thesis Ch. 05
class Optimizer1(Optimization):
    def __init__(self, wf: Workflows, net: LocalNetwork, pb: Problem, report:dict):
        self.var_dim = (wf.num_workflows, wf.num_funs, pb.num_nodes)  # P_n,m,i
        super().__init__(var_dim=self.var_dim)

        self.type = GRB.BINARY
        self.P = self.model.addVars(*self.var_dim, vtype=self.type, name="P")

        self.wf, self.net, self.pb = wf, net, pb

    def setup(self):
        wf, net, pb = self.wf, self.net, self.pb

        # expressions for L, and data transfer costs
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs - 1):
                for node_sending in range(pb.num_nodes):
                    for node_receiving in range(pb.num_nodes):
                        obj_latency = self.P[workflow, function, node_sending] \
                                      * self.P[workflow, function + 1, node_receiving] \
                                      * pb.L[workflow, function, node_sending, node_receiving] \
                                      * wf.funs_counts[workflow, function]
                        self.obj_latency += obj_latency

                        obj_transfer = self.P[workflow, function, node_sending] \
                                       * self.P[workflow, function + 1, node_receiving] \
                                       * pb.D[workflow, function, node_sending, node_receiving] \
                                       * wf.funs_counts[workflow, function]
                        self.obj_transfer += obj_transfer

                        self.obj_latency_details[workflow, function, node_sending, node_receiving] = obj_latency
                        self.obj_transfer_details[workflow, function, node_sending, node_receiving] = obj_transfer

        # expressions for time, and data running costs
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node in range(pb.num_nodes):
                    obj_time = self.P[workflow, function, node] * pb.T[workflow, function, node]\
                               * wf.funs_counts[workflow, function]
                    self.obj_time += obj_time

                    obj_ram = self.P[workflow, function, node] * pb.C[workflow, function, node]\
                              * wf.funs_counts[workflow, function]
                    self.obj_ram += obj_ram

                    obj_startup_cost = pb.C_s[workflow, function, node]
                    self.obj_startup_cost += obj_startup_cost

                    self.obj_time_details[workflow, function, node] = obj_time
                    self.obj_ram_details[workflow, function, node] = obj_ram
                    self.obj_startup_cost_details[workflow, function, node] = obj_startup_cost

        # Constraints

        self.constrain_to_local_nodes(wf=wf, net=net)
        self.constrain_to_cloud_nodes(wf=wf, net=net)

        # Constrain to ram-limit
        for node in range(pb.num_nodes):
            self.model.addConstr(
                quicksum(quicksum(self.P[workflow, function, node] * (wf.funs_data[workflow, function]
                                                                      + wf.funs_sizes[workflow, function])
                                  * wf.funs_counts[workflow, function]
                                  for function in range(wf.num_funs))
                         for workflow in range(wf.num_workflows)) <= pb.ram_limits[node],
                name='ram-limit-constraint')

        # Each function can only be assigned once
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                self.model.addConstr(quicksum(self.P[workflow, function, node] for node in range(pb.num_nodes)) == 1,
                                     name='function-assignment-limit')


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

# Optimizer2 optimizes deployment along edges not nodes to eliminate some non-linear terms
class Optimizer2(Optimization):
    def __init__(self, wf: Workflows, net: LocalNetwork, pb: Problem):
        self.var_dims = (wf.num_workflows, wf.num_funs, pb.num_nodes, pb.num_nodes)  # P_n,m,i,j
        super().__init__(var_dim=self.var_dim)

        self.wf, self.net, self.pb = wf, net, pb

    def setup(self):
        wf, net, pb = self.wf, self.net, self.pb

        # expressions for L, and data transfer costs
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node_sending in range(pb.num_nodes):
                    for node_receiving in range(pb.num_nodes):
                        obj_latency = self.P[workflow, function, node_sending, node_receiving] \
                                      * pb.L[workflow, function, node_sending, node_receiving]
                        self.obj_latency += obj_latency

                        obj_transfer = self.P[workflow, function, node_sending, node_receiving] \
                                       * pb.D[workflow, function, node_sending, node_receiving]
                        self.obj_transfer += obj_transfer

                        self.obj_latency_details[workflow, function, node_sending, node_receiving] = obj_latency
                        self.obj_transfer_details[workflow, function, node_sending, node_receiving] = obj_transfer

        # expressions for time, and data running costs
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node_sending in range(pb.num_nodes):
                    for node_receiving in range(pb.num_nodes):
                        obj_time = self.P[workflow, function, node_sending, node_receiving] \
                                   * pb.T[workflow, function, node_sending]
                        self.obj_time += obj_time

                        obj_ram = self.P[workflow, function, node_sending, node_receiving] \
                                  * pb.C[workflow, function, node_sending]
                        self.obj_ram += obj_ram

                        self.obj_time_details[workflow, function, node_sending, node_receiving] = obj_time
                        self.obj_ram_details[workflow, function, node_sending, node_receiving] = obj_ram

        # Constraints

        # start and end functions should be assigned to TinyFaaS nodes
        # P_n,local,n,j = 1 for all n else 0
        for workflow in range(wf.num_workflows):
            for function in wf.funs_local[workflow]:
                self.model.addConstr(quicksum(self.P[workflow, function, workflow, node_receiving]
                                              for node_receiving in range(pb.num_nodes))
                                     - wf.funs_counts[workflow, function] == 0,
                                     name='local-function-constraint')

        # training functions can not be executed in local nodes
        # sum_k sum_j P_n,train,k,j = 0 , k in {0 .. 3}
        for workflow in range(wf.num_workflows):
            for function in wf.funs_cloud[workflow]:
                self.model.addConstr(quicksum(quicksum(self.P[workflow, function, node_sending, node_receiving]
                                                       for node_sending in range(net.num))
                                              for node_receiving in range(pb.num_nodes)) == 0,
                                     name='cloud-function-constraint')

        # maximum ram limit per sum of functions m on node k
        # sum_n sum_m sum_j P_n,m,i,j * RAM_n,m <= RAM_MAX_i , for all i
        for node_sending in range(pb.num_nodes):
            self.model.addConstr(quicksum(quicksum(quicksum(
                self.P[workflow, function, node_sending, node_receiving]
                * (wf.funs_data[workflow, function] + wf.funs_sizes[workflow, function])
                for node_receiving in range(pb.num_nodes))
                                                   for function in range(wf.num_funs))
                                          for workflow in range(wf.num_workflows)) <= pb.ram_limits[node_sending],
                                 name='ram-limit-constraint')

        # each function can only be assigned to a single node
        # sum_i sum_j P_n,m,i,j = 1 for all n,m
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                self.model.addConstr(quicksum(quicksum(self.P[workflow, function, node_sending, node_receiving]
                                                       for node_receiving in range(pb.num_nodes))
                                              for node_sending in range(pb.num_nodes))
                                     == wf.funs_counts[workflow, function],
                                     name='function-locality-constraint')

        # Flow Conservation Constraint?
        # sum_i P_n,m,i,j = sum_k P_n,m+1,j,k
        # OR: sum_k P_n,m+1,j,k - sum_i P_n,m,i,j = ? (Check Chapter-5: Network Flows)
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs - 1):
                for node_receiving in range(pb.num_nodes):
                    self.model.addConstr(quicksum(self.P[workflow, function, node_sending, node_receiving]
                                                  for node_sending in range(pb.num_nodes))
                                         * wf.funs_counts[workflow, function + 1] ==
                                         quicksum(self.P[workflow, function + 1, node_receiving, node_receiving_next]
                                                  for node_receiving_next in range(pb.num_nodes))
                                         * wf.funs_counts[workflow, function],
                                         name='flow-conservation-constraint?')

    def get_result(self):
        # sum along last axis (sum edges between nodes i, j)
        return np.reshape(np.array([item.x for item in self.model.getVars()]), self.var_dims) \
            .sum(len(self.var_dims) - 1)


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# Optimizer3 formulation failed to reformulate constraints of Optimizer2 for a better parallelization.


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

# Redefine Occupation variables as Optimizer
# Non-linear expressions used
class Optimizer4(Optimization):
    def __init__(self, wf: Workflows, net: LocalNetwork, pb: Problem):
        self.var_dim = (wf.num_workflows, wf.num_funs, pb.num_nodes)  # P_n,m,i
        super().__init__(var_dim=self.var_dim)

        # A variable representing the fraction of function n,m data to be sent along edge i,j
        # d_n,m,i,j := (1 / count_n,m) * Pn,m,i * P_n,m+1,j
        self.d = self.model.addVars(*self.var_dim, pb.num_nodes, vtype=GRB.CONTINUOUS, name="d")

        self.wf, self.net, self.pb = wf, net, pb

    def setup(self):
        wf, net, pb = self.wf, self.net, self.pb

        # Formulate objective function

        self.formulate_objective_loop_nmij(
            expression_nmij=self.formulate_objective_latency_n_transfer)
        self.formulate_objective_loop_nmi(expression_nmi=self.formulate_objective_time_n_ram)
        self.formulate_cost_online_control(wf=wf, pb=pb)

        # Constraints

        self.constrain_to_local_nodes(wf=wf, net=net)
        self.constrain_to_cloud_nodes(wf=wf, net=net)
        self.constrain_to_function_count(wf=wf, pb=pb)
        self.constrain_to_ram_limit_no_temps(wf=wf, pb=pb)
        self.constrain_startup_time(wf=wf, pb=pb)
        # self.constrain_to_ram_limit(wf=wf, pb=pb)
        self.constrain_to_time_limit(wf=wf, pb=pb)
        self.formulate_d()

    def formulate_d(self):
        wf, pb = self.wf, self.pb
        # Non-linear formulation of d_n,m,i,j
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs - 1):
                for node_sending in range(pb.num_nodes):
                    for node_receiving in range(pb.num_nodes):
                        self.model.addConstr(self.d[workflow, function, node_sending, node_receiving]
                                             == (1 / wf.funs_counts[workflow, function])
                                             * self.P[workflow, function, node_sending]
                                             * self.P[workflow, function + 1, node_receiving])

    def formulate_objective_loop_nmij(self, expression_nmij: callable):
        """
        Loop over edges for formulating latency, and data transfer cost. Takes callback functions for execution
        within the loop. Callback functions must return a gurobipy.GenExpr.

        :param expression_nmij: (callable) callback function
               for formulating edge-level latency and data transfer cost objectives
        :return: None
        """

        wf, pb = self.wf, self.pb

        # latency objective is declared as a variable to impose maximum latency constraints
        self.obj_latency_max = self.model.addVars(wf.num_workflows, wf.num_funs, vtype=GRB.CONTINUOUS, name="T_t")

        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs - 1):
                for nd_send in range(pb.num_nodes):
                    for nd_recv in range(pb.num_nodes):
                        index = (workflow, function, nd_send, nd_recv)

                        obj_latency, obj_transfer = expression_nmij(index)

                        self.obj_transfer += obj_transfer

                        self.obj_latency_details[workflow, function, nd_send, nd_recv] = obj_latency
                        self.obj_transfer_details[workflow, function, nd_send, nd_recv] = obj_transfer
                self.obj_latency += self.obj_latency_max[workflow, function]

    def formulate_objective_loop_nmi(self, expression_nmi: callable):
        """
        Loop over edges for formulating time, and data execution costs. Takes callback functions for execution
        within the loop. Callback functions must return a gurobipy.GenExpr.

        :param expression_nmi: (callable) callback function for formulating
               edge-level time and data execution cost objectives
        :return: None
        """

        wf, pb = self.wf, self.pb

        # time objective is declared as a variable to impose maximum time constraints
        self.obj_time_max = self.model.addVars(wf.num_workflows, wf.num_funs, vtype=GRB.CONTINUOUS, name="T")

        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node in range(pb.num_nodes):
                    index = (workflow, function, node)

                    obj_time, obj_ram, obj_startup_cost = expression_nmi(index)

                    self.obj_ram += obj_ram
                    self.obj_startup_cost += obj_startup_cost

                    self.obj_time_details[workflow, function, node] = obj_time + (0 * LinExpr())
                    self.obj_ram_details[workflow, function, node] = obj_ram
                    self.obj_startup_cost_details[workflow, function, node] = obj_startup_cost
                self.obj_time += self.obj_time_max[workflow, function]

    def formulate_objective_latency_n_transfer(self, index: tuple):
        """
        Callback for formulating edge-level latency and data transfer cost objectives (within a loop).
        :param index: (tuple) formulating loop index
        :param pb: (Problem) problem data
        :return gurobipy.LinExpr
        """

        workflow, function = index[0], index[1]

        obj_latency = self.d[index] * self.pb.L[index]
        self.model.addConstr(self.obj_latency_max[workflow, function] >= self.d[index] * self.pb.L[index])

        obj_transfer = self.d[index] * self.pb.D[index]

        return obj_latency, obj_transfer

    def formulate_objective_time_n_ram(self, index: tuple):
        """
        Callback for formulating edge-level time and data execution cost objectives (within a loop).
        :param index: (tuple) formulating loop index
        :return gurobipy.LinExpr
        """
        pb = self.pb
        workflow, function = index[0], index[1]

        obj_time = pb.T[index]
        self.model.addConstr(self.obj_time_max[workflow, function] >= pb.T[index])

        obj_ram = self.P[index] * pb.C[index]

        obj_startup_cost = self.P[index] * pb.C_s[index]

        return obj_time, obj_ram, obj_startup_cost

    def constrain_startup_time(self, wf: Workflows, pb: Problem):
        """
        Create and constrain a variable that pertains to startup time when temporary
        functions overflow local node capacity and have to be loaded from disk.
        See subsection about startup time in chapter 3 of thesis
        :param wf: (Workflows object)
        :param pb: (Problem object)
        :return: None
        """

        self.T_s = self.model.addVars(pb.num_nodes, vtype=GRB.CONTINUOUS)

        # for workflow in range(wf.num_workflows):
        for node in range(pb.num_nodes):
            self.model.addConstr(self.T_s[node] >= pb.taus[node] *
                                 quicksum([self.P[workflow, function, node] * (wf.funs_sizes[workflow, function]
                                                                               + wf.funs_data[workflow, function])
                                           for workflow in range(wf.num_workflows) for function in range(wf.num_funs)])
                                 - pb.ram_limits[node])
            self.model.addConstr(self.T_s[node] >= 0)
            self.obj_startup_time += self.T_s[node]

    def constrain_to_ram_limit_no_temps(self, wf: Workflows, pb: Problem):
        """
        RAM usage is limited by the max-using temporary function in a workflow
        Thus, RAM-limit applies to max usage plus persistent functions
        :param wf: (Workflows object)
        :param pb: (Problem object)
        :return: None
        """

        # A variable for max temporary RAM usage in a workflow
        self.X = self.model.addVars(*(wf.num_workflows, pb.num_nodes))
        for nd in range(pb.num_nodes):
            sum_max_wf_ram = 0
            sum_fn_pers_ram = 0
            for workflow in range(wf.num_workflows):

                for fn_tmp in wf.funs_temp[workflow]:
                    # Maximum RAM-usage per workflow in a node
                    self.model.addConstr(self.X[workflow, nd] >= self.P[workflow, fn_tmp, nd]
                                         * (wf.funs_data[workflow, fn_tmp] + wf.funs_sizes[workflow, fn_tmp]))

                # sum of persistent functions and max workflow ram usage
                for fn_pers in wf.funs_persistent[workflow]:
                    sum_fn_pers_ram += self.P[workflow, fn_pers, nd] \
                                       * (wf.funs_data[workflow, fn_pers] + wf.funs_sizes[workflow, fn_pers])

                # append max workflow RAM usage to persistent functions RAM usage
                sum_max_wf_ram += self.X[workflow, nd]
            self.model.addConstr(sum_max_wf_ram + sum_fn_pers_ram <= pb.ram_limits[nd], name='ram-limit-constraint')

    def formulate_cost_online_control(self, wf: Workflows, pb: Problem):
        # Cost of online-control functions
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                # Check if function is time-limited
                if (workflow, function) in wf.funs_control_rate:
                    processing = quicksum([self.P[workflow, function, nd_f] * pb.C[workflow, function, nd_f]
                                           * pb.T[workflow, function, nd_f] for nd_f in range(pb.num_nodes)])
                    transfer = quicksum([pb.T[workflow, function, nd_f]
                                         * wf.funs_control_rate[workflow, function]\
                                         * ( (self.d[workflow, function, nd_prev, nd_f]
                                              * pb.D[workflow, function, nd_prev, nd_f])
                                             + (self.d[workflow, function, nd_f, nd_prev]
                                                * pb.D[workflow, function, nd_f, nd_prev]) )
                                         for nd_prev in range(pb.num_nodes) for nd_f in range(pb.num_nodes)])
                    cost_control = processing + transfer
                    self.obj_control += cost_control
                    self.obj_control_details[workflow, function] = cost_control

    def constrain_to_time_limit(self, wf: Workflows, pb: Problem):
        # T_f variable is the sum of time to receive, process and send data by f to f-1
        # Assumption!! BOTH f and f-1 are single-instance functions, NOT parallelized
        self.T_f = self.model.addVars(*(wf.num_workflows, pb.num_nodes, pb.num_nodes),
                                      vtype=GRB.CONTINUOUS, name='T_limited')

        # Constrain time-limited functions
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                # Check if function is time-limited
                if (workflow, function) in wf.funs_control_rate:
                    # time_limit = wf.funs_time_limited[(workflow, function)]
                    time_limit = 1 / wf.funs_control_rate[(workflow, function)]
                    for nd_prev in range(pb.num_nodes):  # node of f - 1
                        for nd_f in range(pb.num_nodes):  # node of f
                            latency_in = self.d[workflow, function - 1, nd_prev, nd_f] \
                                         * pb.L[workflow, function - 1, nd_prev, nd_f]
                            latency_out = self.d[workflow, function, nd_f, nd_prev] \
                                          * pb.L[workflow, function, nd_f, nd_prev]
                            time_execution = pb.T[workflow, function, nd_f]
                            total_time = latency_in + time_execution + latency_out
                            # T_f is the total time
                            self.model.addConstr(self.T_f[workflow, nd_prev, nd_f] == total_time)
                            # T_f should be less than time limit
                            self.model.addConstr(self.T_f[workflow, nd_prev, nd_f] <= time_limit)


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

# Linearize d_n,m,i,j from Optimizer4
class Optimizer5(Optimizer4):
    def __init__(self, wf: Workflows, net: LocalNetwork, pb: Problem):
        # Use all properties of Optimizer4
        super().__init__(wf=wf, net=net, pb=pb)

        # Auxiliary variable w_n,m,i,j for linearizing d_n,m,i,j
        # d_n,m,i,j := (1 / count_n,m) * w_n,m,i,j
        self.w = self.model.addVars(*self.var_dim, pb.num_nodes, vtype=GRB.CONTINUOUS, name="w")

    def setup(self):
        wf, net, pb = self.wf, self.net, self.pb

        # Formulate objective function

        self.formulate_objective_loop_nmij(
            expression_nmij=self.formulate_objective_latency_n_transfer)
        self.formulate_objective_loop_nmi(expression_nmi=self.formulate_objective_time_n_ram)
        self.formulate_cost_online_control(wf=wf, pb=pb)

        # Constraints

        self.constrain_to_local_nodes(wf=wf, net=net)
        self.constrain_to_cloud_nodes(wf=wf, net=net)
        self.constrain_to_function_count(wf=wf, pb=pb)
        # self.constrain_to_ram_limit(wf=wf, pb=pb)
        self.constrain_to_ram_limit_no_temps(wf=wf, pb=pb)
        self.constrain_startup_time(wf=wf, pb=pb)
        self.constrain_to_time_limit(wf=wf, pb=pb)
        self.linearize_d()

    def linearize_d(self):
        wf, pb = self.wf, self.pb

        # d_n,m,i,j := (1 / count_n,m) * Pn,m,i * P_n,m+1,j
        # McCormick Relaxation on d_n,mi,j using w as an auxiliary variable
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs - 1):
                for node_sending in range(pb.num_nodes):
                    for node_receiving in range(pb.num_nodes):
                        self.model.addConstr(self.d[workflow, function, node_sending, node_receiving]
                                             == (1 / wf.funs_counts[workflow, function])
                                             * self.w[workflow, function, node_sending, node_receiving])

                        # 1. Upper bound: w <= count_m * P_m+1,j + count_m+1 * P_m,i - count_m * count_m+1
                        self.model.addConstr(self.w[workflow, function, node_sending, node_receiving]
                                             >= (wf.funs_counts[workflow, function]
                                                 * self.P[workflow, function + 1, node_receiving])
                                             + (wf.funs_counts[workflow, function + 1]
                                                * self.P[workflow, function, node_sending])
                                             - (wf.funs_counts[workflow, function]
                                                * wf.funs_counts[workflow, function + 1]))
                        # 2. Lower bound: w >= 0
                        self.model.addConstr(self.w[workflow, function, node_sending, node_receiving] >= 0)
                        # 3. Upper bound: w <= P_m+1,j * count_m
                        self.model.addConstr(self.w[workflow, function, node_sending, node_receiving]
                                             <= wf.funs_counts[workflow, function]
                                             * self.P[workflow, function + 1, node_receiving])
                        # 4. Lower bound: w >= P_m,i * count_m+1
                        self.model.addConstr(self.w[workflow, function, node_sending, node_receiving]
                                             <= wf.funs_counts[workflow, function + 1]
                                             * self.P[workflow, function, node_sending])


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


# Consecutive batches
class Optimizer6(Optimizer5):
    def __init__(self, wf: Workflows, net: LocalNetwork, pb: Problem):
        # Use all properties from Optimizer5
        super().__init__(wf=wf, net=net, pb=pb)

        # formulate batch vars
        self.formulate_batch_vars()

        # flag to indicate that it is batch optimization
        self.flagBatch = True

        # Y-deviation relaxation factor
        self.relaxation_factor = 10

    def setup(self):
        wf, net, pb = self.wf, self.net, self.pb

        # Formulate objective function

        self.formulate_objective_loop_nmij(
            expression_nmij=self.formulate_objective_latency_n_transfer)
        self.formulate_objective_loop_nmi(expression_nmi=self.formulate_objective_time_n_ram)
        self.formulate_cost_online_control(wf=wf, pb=pb)

        # Constraints

        self.constrain_to_local_nodes(wf=wf, net=net)
        self.constrain_to_cloud_nodes(wf=wf, net=net)
        self.constrain_to_function_count(wf=wf, pb=pb)
        self.constrain_to_ram_limit_no_temps(wf=wf, pb=pb)
        self.constrain_startup_time(wf=wf, pb=pb)
        # self.constrain_to_ram_limit(wf=wf, pb=pb)
        self.constrain_to_time_limit(wf=wf, pb=pb)
        self.formulate_d()
        self.constrain_batch_vars_nonlinear(wf=wf, pb=pb)
        self.constrain_y_deviation(wf=wf, net=net, pb=pb)

    # Add startup time as an objective. see section 3.5.3 in thesis
    def constrain_startup_time(self, wf: Workflows, pb: Problem):
        """
        Create and constrain a variable that pertains to startup time when temporary
        functions overflow local node capacity and have to be loaded from disk.
        See subsection about startup time in chapter 3 of thesis
        :param wf: (Workflows object)
        :param pb: (Problem object)
        :return: None
        """

        self.T_s = self.model.addVars(pb.num_nodes, vtype=GRB.CONTINUOUS)

        # for workflow in range(wf.num_workflows):
        for node in range(pb.num_nodes):
            self.model.addConstr(self.T_s[node] >= pb.taus[node] *
                                 quicksum( [ self.P[workflow, function, node] * (wf.funs_sizes[workflow, function]
                                                                                 + wf.funs_data[workflow, function])
                                             for workflow in range(wf.num_workflows) for function in range(wf.num_funs)
                                             if wf.funs_counts[workflow, function] <= 1] ) - pb.ram_limits[node] )
            self.model.addConstr(self.T_s[node] >= pb.taus[node] *
                                 quicksum([self.x[workflow, function, node] * (wf.funs_sizes[workflow, function]
                                                                               + wf.funs_data[workflow, function])
                                           for workflow in range(wf.num_workflows) for function in range(wf.num_funs)
                                           if wf.funs_counts[workflow, function] > 1]) - pb.ram_limits[node] )

            self.model.addConstr(self.T_s[node] >= 0)
            self.obj_startup_time += self.T_s[node]

    def constrain_to_ram_limit_no_temps(self, wf: Workflows, pb: Problem):
        # RAM usage is limited by the max-using temporary function in a workflow
        # Thus, RAM-limit applies to max usage plus persistent functions
        # A variable for max temporary RAM usage in a workflow
        self.R = self.model.addVars(*(wf.num_workflows, pb.num_nodes))

        for node in range(pb.num_nodes):
            sum_max_wf_ram = 0
            sum_fn_pers_ram = 0
            for workflow in range(wf.num_workflows):

                for fn_tmp in wf.funs_temp[workflow]:
                    # check if function is parallelizable
                    if wf.funs_counts[workflow, fn_tmp] <= 1:
                        # Maximum RAM-usage per workflow in a node
                        self.model.addConstr(self.R[workflow, node] >= self.P[workflow, fn_tmp, node]
                                             * (wf.funs_data[workflow, fn_tmp] + wf.funs_sizes[workflow, fn_tmp]))
                    else:  # if parallelizable, then the number of parallel instances 'x' is the limiting variable
                        self.model.addConstr(self.R[workflow, node] >= self.x[workflow, fn_tmp, node]
                                             * (wf.funs_data[workflow, fn_tmp] + wf.funs_sizes[workflow, fn_tmp]))

                # sum of persistent functions and max workflow ram usage
                for fn_pers in wf.funs_persistent[workflow]:
                    # persistent functions are not parallelizable
                    sum_fn_pers_ram += self.P[workflow, fn_pers, node] \
                                       * (wf.funs_data[workflow, fn_pers] + wf.funs_sizes[workflow, fn_pers])

                # append max workflow RAM usage to persistent functions RAM usage
                sum_max_wf_ram += self.R[workflow, node]
            self.model.addConstr(sum_max_wf_ram + sum_fn_pers_ram <= pb.ram_limits[node],
                                 name='ram-limit-constraint')

    def constrain_to_ram_limit(self, wf: Workflows, pb: Problem):
        # RAM usage is limited by the total assigned functions
        for node in range(pb.num_nodes):
            # if function is not parallelizable
            self.model.addConstr(
                quicksum(self.P[workflow, function, node] * (wf.funs_data[workflow, function]
                                                             + wf.funs_sizes[workflow, function])
                         for function in range(wf.num_funs) for workflow in range(wf.num_workflows)
                         if wf.funs_counts[workflow, function] <= 1) <= pb.ram_limits[node],
                name='ram-limit-constraint_xy1')
            # if parallelizable, then the number of parallel instances 'x' is the limiting variable
            self.model.addConstr(
                quicksum(self.x[workflow, function, node] * (wf.funs_data[workflow, function]
                                                             + wf.funs_sizes[workflow, function])
                         for function in range(wf.num_funs) for workflow in range(wf.num_workflows)
                         if wf.funs_counts[workflow, function] > 1) <= pb.ram_limits[node],
                name='ram-limit-constraint_xyc')

    def constrain_batch_vars_nonlinear(self, wf: Workflows, pb: Problem):
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node in range(pb.num_nodes):
                    self.model.addConstr(self.P[workflow, function, node] ==
                                         self.x[workflow, function, node] * self.y[workflow, function, node])

    def constrain_y_deviation(self, wf: Workflows, net: LocalNetwork, pb: Problem):
        # collect sets of different p_factors
        p_factors_set = set(pb.p_factors)
        p_factors_nodes = {p.item(): np.argwhere(p == pb.p_factors).flatten().tolist() for p in p_factors_set}
        relaxation_factor = self.relaxation_factor
        node_firsts_list = []
        worst_efficiency = min(p_factors_set)
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                # nodes with same p-factor should have same y
                for p in p_factors_set:
                    # exit loop if p_factor is only for a single node
                    if len(p_factors_nodes[p]) <= 1:
                        break
                    # index of first node in p_factor node list
                    node_first = p_factors_nodes[p][0]
                    node_firsts_list.append(node_first)

                    for node in p_factors_nodes[p][1:]:
                        # y should be the same for similar p-factor nodes
                        self.model.addConstr(self.y[workflow, function, node_first]
                                             == self.y[workflow, function, node])

                # difference between nodes of different speeds should not exceed certain factor
                slowest_node = p_factors_nodes[worst_efficiency][0]
                for node in node_firsts_list:
                    if not node == slowest_node:
                        self.model.addConstr(
                            (self.y[workflow, function, slowest_node] * pb.T[workflow, function, slowest_node].item()
                             - self.y[workflow, function, node] * pb.T[workflow, function, node].item())
                            <= (relaxation_factor * (pb.T[workflow, function, slowest_node].item()
                                                     - pb.T[workflow, function, node].item())))

    def formulate_batch_vars(self):
        wf, pb = self.wf, self.pb

        # variables for batch processing within a node:
        # x: number of parallel instances, y: number of consecutive batches
        self.x = {}
        self.y = {}
        # breakpoints for piecewise optimization of x, y choices
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node in range(pb.num_nodes):
                    if wf.funs_counts[workflow, function] <= 1:
                        self.x[workflow, function, node] = self.P[workflow, function, node]
                        self.y[workflow, function, node] = 1
                    else:
                        var_name = f'wf{workflow}_f{function}_n{node}'
                        self.x[workflow, function, node] = self.model.addVar(vtype=GRB.INTEGER, lb=0,
                                                                             ub=wf.funs_counts[workflow, function],
                                                                             name='x_' + var_name)
                        self.y[workflow, function, node] = self.model.addVar(vtype=GRB.INTEGER, lb=0,
                                                                             ub=wf.funs_counts[workflow, function],
                                                                             name='y_' + var_name)

    def formulate_objective_time_n_ram(self, index: tuple):
        """
        Callback for formulating edge-level time and data execution cost objectives (within a loop).
        :param index: (tuple) formulating loop index
        :return gurobipy.LinExpr
        """

        workflow, function = index[0], index[1]
        wf, pb = self.wf, self.pb

        if wf.funs_counts[workflow, function] <= 1:
            self.model.addConstr(self.obj_time_max[workflow, function] >= pb.T[index])
            obj_time = pb.T[index]
            obj_ram = self.P[index] * pb.C[index]
            obj_startup_cost = self.P[index] * pb.C_s[index]
        else:
            self.model.addConstr(self.obj_time_max[workflow, function] >=
                                 pb.T[index] * self.y[index])
            obj_time = pb.T[index] * self.y[index]

            obj_ram = self.P[index] * pb.C[index]
            obj_startup_cost = self.x[index] * pb.C_s[index]

        return obj_time, obj_ram, obj_startup_cost


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


# Function to generate breakpoints using a hybrid logarithmic-linear scale
def generate_breakpoints(max_value, num_log=10, num_lin=10):
    """
    Generates breakpoints using a hybrid logarithmic-linear scale.

    Parameters:
        max_value (int): The upper bound for x and y.
        num_log (int): Number of log-scale points.
        num_lin (int): Number of linear-scale points.

    Returns:
        list: Sorted list of breakpoints.
    """
    log_space = np.logspace(0, np.log10(max_value / 10), num=num_log, base=10, dtype=int)
    lin_space = np.linspace(log_space[-1], max_value, num=num_lin, dtype=int)

    breakpoints = sorted(set(log_space.tolist() + lin_space.tolist()))  # Ensure uniqueness & sorting
    breakpoints.insert(0, 0)
    return breakpoints


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


# Consecutive batches linearized
class Optimizer7(Optimizer6):
    def __init__(self, wf: Workflows, net: LocalNetwork, pb: Problem):
        # Use all properties from Optimizer5
        super().__init__(wf=wf, net=net, pb=pb)

        self.wf, self.net, self.pb = wf, net, pb

        # number of discretization points for x and y
        self. n_breakpnts = 20

        # Y-depth shortening factor
        self.depth_factor = 10

    def setup(self):
        wf, net, pb = self.wf, self.net, self.pb

        self.linearize_batch_vars()

        # Formulate objective function

        self.formulate_objective_loop_nmij(
            expression_nmij=self.formulate_objective_latency_n_transfer)
        self.formulate_objective_loop_nmi(expression_nmi=self.formulate_objective_time_n_ram)
        self.formulate_cost_online_control(wf=wf, pb=pb)

        # Constraints

        self.constrain_to_local_nodes(wf=wf, net=net)
        self.constrain_to_cloud_nodes(wf=wf, net=net)
        self.constrain_to_function_count(wf=wf, pb=pb)
        self.constrain_to_ram_limit_no_temps(wf=wf, pb=pb)
        self.constrain_startup_time(wf=wf, pb=pb)
        # self.constrain_to_ram_limit(wf=wf, pb=pb)
        self.constrain_to_time_limit(wf=wf, pb=pb)
        self.constrain_y_deviation(wf=wf, net=net, pb=pb)
        self.linearize_d()

    def linearize_batch_vars(self):
        wf, pb = self.wf, self.pb

        # Auxiliary variable for linearization constraints
        self.P_pw = {}  # piecewise optimized P
        # breakpoints for piecewise optimization of x, y choices
        self.x_vals = {}
        self.y_vals = {}
        self.p_vals = {}
        self.lambda_vars = {}
        for workflow in range(wf.num_workflows):
            for function in range(wf.num_funs):
                for node in range(pb.num_nodes):
                    if wf.funs_counts[workflow, function] <= 1:
                        self.x[workflow, function, node] = self.P[workflow, function, node]
                        self.y[workflow, function, node] = 1
                    else:
                        # breakpoint generation
                        # different bounds for x, y for tighter constraint
                        self.x_vals[workflow, function, node] = \
                            generate_breakpoints(max_value=wf.funs_counts[workflow, function],
                                                 num_log=self.n_breakpnts, num_lin=self.n_breakpnts)
                        self.y_vals[workflow, function, node] = \
                            generate_breakpoints(max_value=np.ceil(wf.funs_counts[workflow, function] / self.depth_factor),
                                                 num_log=self.n_breakpnts, num_lin=self.n_breakpnts)
                        # Compute P = x * y at breakpoints
                        self.p_vals[workflow, function, node] = np.array(
                            [[xi * yi for yi in self.y_vals[workflow, function, node]]
                             for xi in self.x_vals[workflow, function, node]])

                        var_name = f'wf{workflow}_f{function}_n{node}'
                        self.P_pw[workflow, function, node] = self.model.addVar(vtype=GRB.INTEGER,
                                                                                name="F_pw_" + var_name)

                        # P_n,m,i = P_pw to enforce correct multiplication
                        self.model.addConstr(self.P[workflow, function, node] == self.P_pw[workflow, function, node])

                        # Interpolation variables for breakpoints
                        self.lambda_vars[workflow, function, node] = \
                            self.model.addVars(len(self.x_vals[workflow, function, node]),
                                               len(self.y_vals[workflow, function, node]),
                                               vtype=GRB.BINARY, name="lambda")

                        # Ensure lambda forms a convex combination
                        self.model.addConstr(quicksum(self.lambda_vars[workflow, function, node][i, j]
                                                      for i in range(len(self.x_vals[workflow, function, node]))
                                                      for j in range(len(self.y_vals[workflow, function, node]))) == 1,
                                             name="lambda_sum")

                        # Express x, y, and P as convex combinations
                        self.model.addConstr(self.x[workflow, function, node] ==
                                             quicksum(self.lambda_vars[workflow, function, node][i, j] *
                                                      self.x_vals[workflow, function, node][i]
                                                      for i in range(len(self.x_vals[workflow, function, node]))
                                                      for j in range(len(self.y_vals[workflow, function, node]))),
                                             name="x_interp")
                        self.model.addConstr(self.y[workflow, function, node] ==
                                             quicksum(self.lambda_vars[workflow, function, node][i, j] *
                                                      self.y_vals[workflow, function, node][j]
                                                      for i in range(len(self.x_vals[workflow, function, node]))
                                                      for j in range(len(self.y_vals[workflow, function, node]))),
                                             name="y_interp")
                        self.model.addConstr(self.P_pw[workflow, function, node] == quicksum(
                            self.lambda_vars[workflow, function, node][i, j]
                            * self.p_vals[workflow, function, node][i, j]
                            for i in range(len(self.x_vals[workflow, function, node]))
                            for j in range(len(self.y_vals[workflow, function, node]))), name="P_interp")

                        # SOS2 constraint: At most two adjacent lambda variables are nonzero
                        # Mostly not needed, refer to conversation with chatGPT to know more
                        # for i in range(len(self.x_vals[workflow, function, node])):
                        #     self.model.addSOS(GRB.SOS_TYPE2, [self.lambda_vars[workflow, function, node][i, j]
                        #                                       for j in
                        #                                       range(len(self.y_vals[workflow, function, node]))])
                        # for j in range(len(self.y_vals[workflow, function, node])):
                        #     self.model.addSOS(GRB.SOS_TYPE2, [self.lambda_vars[workflow, function, node][i, j]
                        #                                       for i in
                        #                                       range(len(self.x_vals[workflow, function, node]))])

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
