import json
import os.path


class Workflows:
    def __init__(self, datapath: str):
        # import data from JSON file
        with open(os.path.abspath(datapath), "r") as f:
            workflows = json.load(f)

        # Extract workflows and functions' properties
        self.num_workflows = len(workflows)
        self.names_workflows = list(workflows.keys())
        self.names_funs = {}
        self.funs_times = {}
        self.funs_data = {}
        self.funs_sizes = {}
        self.funs_counts = {}
        self.funs_local = {}
        self.funs_cloud = {}
        self.funs_persistent = {}
        self.funs_temp = {}
        self.funs_time_limited = {}
        self.funs_control_rate = {}
        num_workflow_funs = []

        for wi, workflow in enumerate(workflows.values()):
            self.names_funs[wi] = []
            self.funs_local[wi] = []
            self.funs_cloud[wi] = []
            self.funs_persistent[wi] = []
            self.funs_temp[wi] = []

            # extract workflow function information
            for fi, function in enumerate(workflow["functions"]):
                self.names_funs[wi].append(function)
                fun_info = workflow["functions"][function]
                self.funs_times[wi, fi] = fun_info["time"]  # second
                self.funs_data[wi, fi] = fun_info["data"]  # GB
                self.funs_sizes[wi, fi] = fun_info["size"]  # GB
                self.funs_counts[wi, fi] = fun_info["count"]  # number of function instances
                if fun_info["location"] == "local":
                    self.funs_local[wi].append(fi)
                elif fun_info["location"] == "cloud":
                    self.funs_cloud[wi].append(fi)
                if fun_info["run"] == "always":
                    self.funs_persistent[wi].append(fi)
                elif fun_info["run"] == "temp":
                    self.funs_temp[wi].append(fi)
                if "time_limit" in fun_info:
                    self.funs_time_limited[(wi, fi)] = fun_info["time_limit"]
                if "control_rate" in fun_info:
                    self.funs_control_rate[(wi, fi)] = fun_info["control_rate"]

            # store the number of functions in a workflow
            num_workflow_funs.append(len(workflow["functions"].keys()))

        # Pad workflows in case of varying number of functions
        # get the number of functions from the maximum length workflow
        self.num_funs = max(num_workflow_funs)
        # Check if not all workflows have same number of functions
        if len(set(num_workflow_funs)) != 1:
            for wi, workflow in enumerate(self.names_funs.values()):
                workflow_size = len(workflow)
                if workflow_size < self.num_funs:
                    for gap in range(self.num_funs - workflow_size):
                        self.names_funs[wi].append("dummy")
                        self.funs_times[wi, gap].append(0)
                        self.funs_data[wi, gap].append(0)
                        self.funs_sizes[wi, gap].append(0)
                        self.funs_counts[wi, gap].append(1)
                        self.funs_local[wi].append(workflow_size + gap)


class LocalNetwork:
    def __init__(self, datapath: str):
        # import data from JSON file
        with open(os.path.abspath(datapath), "r") as f:
            network_config = json.load(f)

        # Extract network properties
        self.latency_local = network_config["latency"]  # second
        self.num = len(network_config["nodes"])
        self.rams = []  # available runtime memory
        self.p_factors = []  # parallelization efficiency
        self.taus = []  # startup time factor [s/GB]
        self.rate_up = network_config["rate_up"]  # upload speed [GB/s]
        self.rate_down = network_config["rate_down"]  # download speed [GB/s]
        for node in network_config["nodes"]:
            self.rams.append(network_config["nodes"][node]["ram"])  # GB
            self.p_factors.append(network_config["nodes"][node]["p_factor"])  # parallelization efficacy factor
            self.taus.append(network_config["nodes"][node]["tau"])  # startup time factor


def check_network_validity(num_workflows: int, num_tiny: int):
    # Check if available nodes are enough for workflows
    if num_workflows > num_tiny:
        raise ValueError(f"TinyFaaS nodes ({num_tiny} are not enough for {num_workflows} workflows)")


class CloudsInfo:
    def __init__(self, datapath: str):
        # import data from JSON file
        with open(os.path.abspath(datapath), "r") as f:
            cloud_config = json.load(f)

        # Extract cloud providers' information
        self.names = list(cloud_config.keys())
        self.num = len(self.names)
        self.prices_transfer_up = []
        self.prices_transfer_down = []
        self.prices_ram = []
        self.prices_start = []
        self.latency = []
        self.p_factors = []
        self.taus = []
        for provider in cloud_config.values():
            self.prices_transfer_up.append(provider["pricing_Storage_Transfer"]["upload"])  # $/GB
            self.prices_transfer_down.append(provider["pricing_Storage_Transfer"]["download"])  # $/GB
            self.prices_ram.append(provider["pricing_RAM"])  # GB*s
            self.prices_start.append(provider["pricing_StartRequest"])  # $/request
            self.latency.append(provider["estimated_latency"])  # second
            self.p_factors.append(provider["p_factor"])  # parallelization efficacy factor
            self.taus.append(provider["tau"])  # startup time factor in second per GB
