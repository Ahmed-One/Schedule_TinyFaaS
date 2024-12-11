from gurobipy import *
from creation import *

model = Model("TinyFaaS")
P = model.addVars(num_tiny, num_funs, num_nodes, vtype=GRB.BINARY, name="P")  # P_n,m,i

# expressions for latency, and data transfer costs
obj_latency = 0
obj_transfer = 0
for workflow in range(num_tiny):
    for function in range(num_funs - 1):
        for node_sending in range(num_nodes):
            for node_receiving in range(num_nodes):
                obj_latency = obj_latency + P[workflow, function, node_sending] \
                              * P[workflow, function + 1, node_receiving] \
                              * L[node_sending, node_receiving]
                obj_transfer = obj_transfer + P[workflow, function, node_sending] \
                               * P[workflow, function + 1, node_receiving] \
                               * D[workflow, function, node_sending, node_receiving]

# expressions for time, and data running costs
obj_time = 0
obj_ram = 0
for workflow in range(num_tiny):
    for function in range(num_funs):
        for node in range(num_nodes):
            obj_time = obj_time + P[workflow, function, node] * T[workflow, function, node]
            obj_ram = obj_ram + P[workflow, function, node] * C[workflow, function, node]

# expression for the total objective function
w_1, w_2 = 0.5, 0.3  # w1 for latency and time, w2 for costs
obj = w_1 * (obj_latency + obj_time) + w_2 * (obj_transfer + obj_ram)

# Constraints

# start and end functions should be assigned to TinyFaaS nodes
# P_n,(0,5,7),n = 1 for all n else 0
for workflow in range(num_tiny):
    for function in start_ends:
        for node in range(num_nodes):
            if node >= num_tiny:
                model.addConstr(P[workflow, function, node] == 0)

# training functions can not be executed in local nodes
# sum_k P_n,4,k = 0 , k in {0 .. 3}
for workflow in range(num_tiny):
    for function in funs_cloud:
        model.addConstr(quicksum(P[workflow, function, node] for node in range(num_tiny)) == 0)

# maximum ram limit per sum of functions m on node k
# sum_n sum_k P_n,m,k * RAM_n,m <= MAX_k , for all k in {0 ..3} for all n, m
for workflow in range(num_tiny):
    for node in range(num_nodes):
        model.addConstr(quicksum(P[workflow, function, node] * rams[function]
                                 for function in range(num_funs)) <= ram_limits[node])

# each function can only be assigned to a single node
#
for workflow in range(num_tiny):
    for function in range(num_funs):
        model.addConstr(quicksum(P[workflow, function, node] for node in range(num_nodes)) == 1)

# Solve model
model.setObjective(obj, GRB.MINIMIZE)
model.setParam("OutputFlag", 1)
model.update()
model.optimize()

# Print result
P_out = np.reshape(np.array([item.x for item in model.getVars()]), (num_tiny, num_funs, num_nodes))
print(P_out)
