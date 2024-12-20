# Script to configure the JSON files with random values
import json
import os.path
import random

random.seed()
workflows_dict = {}

# Workflows property
wf_count = (1, 3)

# Function property ranges
time = (0, 1)  # seconds
data = (0, 1)  # megabytes
size = (0, 1)  # megabytes
count = (10, 1000)
location_local = "local"
location_cloud = "cloud"
location_any = "any"


# function to create workflows from parameters
def configure_wf_dict(wf_dict, wf_funcs, wf_name="test", wf_num=1):
    for workflow in range(wf_num):
        workflows_dict[f"workflow-{wf_name}{workflow}"] = {"functions": {}}
        for function in wf_funcs:

            location_f = location_any
            time_f = round(random.uniform(*time), 3)
            count_f = 1
            data_f = round(random.uniform(*data), 3)
            size_f = round(random.uniform(*size), 3)

            if "start" in function or "end" in function or "control" in function:
                location_f = location_local
                time_f = 0
                data_f = 0
                size_f = 0
            elif "train" in function:
                location_f = location_cloud

                # function should have multiple instances only when it is "generate_samples"
            if any(x in function for x in ["generate", "samples"]) and not("init" in function):
                count_f = random.randint(*count)

            function_dict = {
                "location": location_f,
                "time": time_f,
                "data": data_f,
                "size": size_f,
                "count": count_f
            }

            wf_dict[f"workflow-{wf_name}{workflow}"]["functions"][function] = function_dict


# create test workflows
wf_test = ["start", "generate_init_samples", "generate_samples", "train", "end"]
configure_wf_dict(workflows_dict, wf_test, wf_name="test", wf_num=1)

# create tinyFaaS workflows
wf_tiny = ["start", "generate_init_samples", "generate_samples", "train", "end_train",
           "control", "end_control"]
configure_wf_dict(workflows_dict, wf_tiny, wf_name="tiny", wf_num=3)

with open(os.path.abspath("requirements//workflows2.json"), "w+") as f:
    json.dump(workflows_dict, f, indent=2)
