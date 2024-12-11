# Create the coefficient matrices
import numpy as np

# Simple problem
# 2 local nodes, 2 clouds, 4-function workflows

# Functions
# 0: start, 1: inits, 2: train, 3: load, 4:end
names_funs = ['start', 'inits', 'train', 'load', 'end']
start_ends = [0, 4]
funs_cloud = [2]  # index of functions that must be run on cloud
times = [0, 3, 8, 4, 0]
rams = [0, 2, 12, 3, 0]
num_funs = len(times)

# TinyFaaS nodes (2-nodes/workflows)
names_tiny = ['0', '1']
rams_tiny = [5, 3]
num_tiny = len(names_tiny)

# Clouds (2-providers)
names_cloud = ['G', 'A']
prices_cloud_transfer = [10, 12]
prices_cloud_ram = [12, 10]
prices_cloud_start = [20, 22]
num_clouds = len(names_cloud)

# some helpful data
num_nodes = num_tiny + num_clouds
prices_transfer = np.append(np.zeros(num_tiny), prices_cloud_transfer)
prices_ram = np.append(np.zeros(num_tiny), prices_cloud_ram)
prices_start = np.append(np.zeros(num_tiny), prices_cloud_start)
ram_limits = np.append(np.array(rams_tiny), np.inf * np.ones(num_clouds))

# Latency
# TinyFaaS: 0, 1, Cloud: 2, 3
latency = [[0, 0.01, 0.1, 0.2],
           [0.01, 0, 0.2, 0.1],
           [0.1, 0.2, 0, 0.3],
           [0.2, 0.1, 0.3, 0]]

# Coefficient matrices
# Latency between nodes
L = np.array(latency)  # L_j,i

# Time for running a function on a node
T = np.zeros((num_tiny, num_funs, num_nodes))  # T_n,m,j
for workflow in range(num_tiny):
    for node in range(num_nodes):
        T[workflow, :, node] = np.array(times)

# Cost of transferring data
D = np.zeros((num_tiny, num_funs, num_nodes, num_nodes))  # D_n,m,j,i
for workflow in range(num_tiny):
    for function in range(num_funs):
        for node_sending in range(num_nodes):
            for node_receiving in range(num_nodes):
                D[workflow, function, node_sending, node_receiving] = rams[function] * prices_ram[node_sending] + \
                                                                      rams[function] * prices_ram[node_receiving]
# Cost of running a function
C = np.zeros((num_tiny, num_funs, num_nodes))  # C_n,m,j
for workflow in range(num_tiny):
    for function in range(num_funs):
        for node in range(num_nodes):
            C[workflow, function, node] = prices_start[node] + rams[function] * prices_ram[node]

pass
