import json
import os.path


class Workflows:
    def __init__(self, datapath: str):
        # import data from JSON file
        with open(os.path.abspath(datapath), "r") as f:
            workflows = json.load(f)

        # Extract workflows and functions' properties
        self.num_workflows = 0
        self.names_workflows = []
        self.names_funs = []
        self.funs_local = []
        self.funs_cloud = []
        self.funs_times = []
        self.funs_data = []
        self.funs_sizes = []
        self.funs_counts = []
        num_workflow_funs = []

        for workflow in workflows:
            self.names_workflows.append(workflow)

        for workflow in workflows.values():
            self.num_workflows += 1
            fun_count = []
            names = []
            times = []
            datas = []
            sizes = []
            start_ends = []
            clouds = []
            index = 0
            # extract workflow function information
            for index, function in enumerate(workflow["functions"]):
                names.append(function)
                fun_info = workflow["functions"][function]
                times.append(fun_info["time"])  # second
                datas.append(fun_info["data"])  # GB
                sizes.append(fun_info["size"])  # GB
                if fun_info["location"] == "local":
                    start_ends.append(index)
                elif fun_info["location"] == "cloud":
                    clouds.append(index)
                fun_count.append(fun_info["count"])  # test

            # store the number of functions in a workflow
            num_workflow_funs.append(index + 1)

            self.names_funs.append(names)
            self.funs_times.append(times)
            self.funs_data.append(datas)
            self.funs_sizes.append(sizes)
            self.funs_local.append(start_ends)
            self.funs_cloud.append(clouds)
            self.funs_counts.append(fun_count)

        # Pad workflows in case of varying number of functions
        # get the number of functions from the maximum length workflow
        self.num_funs = max(num_workflow_funs)
        # Check if not all workflows have same number of functions
        if len(set(num_workflow_funs)) != 1:
            for index, workflow in enumerate(self.names_funs):
                workflow_size = len(workflow)
                if workflow_size < self.num_funs:
                    for gap in range(self.num_funs - workflow_size):
                        self.names_funs[index].append("dummy")
                        self.funs_times[index].append(0)
                        self.funs_data[index].append(0)
                        self.funs_sizes[index].append(0)
                        self.funs_counts[index].append(1)
                        self.funs_local[index].append(workflow_size + gap)


class LocalNetwork:
    def __init__(self, datapath: str):
        # import data from JSON file
        with open(os.path.abspath(datapath), "r") as f:
            network_config = json.load(f)

        # Extract network properties
        self.latency_local = network_config["latency"]  # second
        self.num_tiny = len(network_config["nodes"])
        self.rams_tiny = []
        for node in network_config["nodes"]:
            self.rams_tiny.append(network_config["nodes"][node])  # GB


def check_network_validity(num_workflows: int, num_tiny: int):
    # Check if available nodes are enough for workflows
    if num_workflows > num_tiny:
        raise ValueError(f"TinyFaaS nodes ({num_tiny} are nor enough for {num_workflows} workflows)")


class CloudsInfo:
    def __init__(self, datapath: str):
        # import data from JSON file
        with open(os.path.abspath(datapath), "r") as f:
            cloud_config = json.load(f)

        # Extract cloud providers' information
        self.names_cloud = cloud_config.keys()
        self.num_clouds = len(self.names_cloud)
        self.prices_cloud_transfer = []
        self.prices_cloud_ram = []
        self.prices_cloud_start = []
        self.latency_cloud = []
        for provider in cloud_config.values():
            self.prices_cloud_transfer.append(["pricing_Storage_Transfer"])  # $/GB
            self.prices_cloud_ram.append(provider["pricing_RAM"])  # GB*s
            self.prices_cloud_start.append(provider["pricing_StartRequest"])  # $/request
            self.latency_cloud.append(provider["estimated_latency"])  # second
