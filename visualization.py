# Visualize optimization results
import pandas as pd
from dataclasses import dataclass, field
import numpy as np

from parsing import *


# Helper function for abbreviating names
def abbreviate_fun_names(functions: list):
    for index, function in enumerate(functions):
        if "generate" in function:
            functions[index] = functions[index].replace("generate", "gnrt")
        if "samples" in function:
            functions[index] = functions[index].replace("samples", "smpl")
        if "dummy" in function:
            functions[index] = functions[index].replace("dummy", "dmy")


def abbreviate_wf_name(workflow_name: str):
    if "workflow" in workflow_name:
        workflow_name.replace("workflow", "wf")
    if "test" in workflow_name:
        workflow_name.replace("test", "tst")
    if "tiny" in workflow_name:
        workflow_name.replace("tiny", "tny")


@dataclass
class WorkflowTable:
    name: str
    functions: list
    nodes: list
    p_matrix: np.ndarray
    table: pd.DataFrame = field(init=False)

    def __post_init__(self):
        # Abbreviate function names
        abbreviate_fun_names(self.functions)

        # Add column of function names to the occupation matrix
        entries = self.p_matrix.tolist()
        for index, function in enumerate(self.functions):
            entries[index].insert(0, function)

        # Augment table header with "functions" title
        headers = ["functions", *self.nodes]

        # Create pandas dataframe
        self.table = pd.DataFrame(entries, columns=headers)
        self.table.set_index("functions")

    def print(self):
        print("_" * len(self.name))
        print(self.name)
        pd.set_option('display.max_colwidth', None)
        print(self.table)


# Use Pandas library
class WorkflowsTable:
    def __init__(self, wf: Workflows, cld: CloudsInfo, P: np.ndarray):
        self.workflow_tables = []
        for index, workflow in enumerate(wf.names_workflows):
            name = workflow
            names_cloud = list(cld.names_cloud)
            names_funs = wf.names_funs[index]
            num_nodes = P.shape[2]  # number of nodes (0: wf x 1: fun x 3: nodes)
            num_tiny = num_nodes - len(names_cloud)
            nodes = [f"Nd-{i}" for i in range(num_tiny)]
            [nodes.append(cloud_name) for cloud_name in names_cloud]
            self.workflow_tables.append(WorkflowTable(name=name, functions=names_funs, nodes=nodes, p_matrix=P[index]))

    def print(self):
        for table in self.workflow_tables:
            table.print()


class VisualUML:
    def __init__(self, wf: Workflows, net: LocalNetwork, cld: CloudsInfo, P: np.ndarray):
        # Create dictionary to hold information about occupiers of each node
        self.nodes = {f"Node_{i}": {} for i in range(net.num_tiny)}
        self.nodes.update({cloud_name: {} for cloud_name in cld.names_cloud})
        self.num_tiny = net.num_tiny

        for i_node, node in enumerate(self.nodes):
            # Assign node ram limit
            if i_node < net.num_tiny:
                self.nodes[node]["ram_limit"] = net.rams_tiny[i_node]
            else:
                self.nodes[node]["ram_limit"] = np.inf

            self.nodes[node]["ram_used"] = 0
            self.nodes[node]["workflows"] = {}
            for i_wf, workflow in enumerate(wf.names_workflows):
                # Check if any function within current workflow is deployed to this node
                if sum(abs(P[i_wf, :, i_node])) > 0:
                    self.nodes[node]["workflows"][workflow] = []
                    for i_f, function in enumerate(wf.names_funs[i_wf]):
                        # Assign used ram
                        self.nodes[node]["ram_used"] += round(P[i_wf, i_f, i_node] * wf.funs_data[i_wf][i_f])
                        # Assign deployed function(s)
                        if abs(P[i_wf, i_f, i_node]) > 0:
                            self.nodes[node]["workflows"][workflow].append(f"{function}: "
                                                                           f"{round(P[i_wf, i_f, i_node])}"
                                                                           f"/{wf.funs_counts[i_wf][i_f]}")

    def code_a_node(self, node: str):
        node_data = self.nodes[node]
        node_code = [
            f"<b>{node}",  # node name in bold
            "===="  # separator
        ]
        for workflow in node_data["workflows"]:
            node_code.append(f"{workflow}")  # workflow name
            node_code.append("....")  # separator
            for function in node_data["workflows"][workflow]:
                node_code.append(function)
            node_code.append("----")
        node_code.append(f"RAM usage:    {node_data['ram_used']}/{node_data['ram_limit']}")
        return node_code

    def code_diagram(self):
        uml_code = ["@startuml", "\n"]  # first line in a 'uml' file

        for i, node in enumerate(self.nodes):
            # declare a deployment diagram entity based on node type
            if i < self.num_tiny:
                uml_code.append(f"node {node}[")
            else:
                uml_code.append(f"cloud {node}[")

            # fill in node information coded with indentation
            for line in self.code_a_node(node):
                uml_code.append("  " + line)

            uml_code.append("]")  # close node description

        uml_code.append("\n")  # empty line after node description

        # connect nodes with arrows
        for i, node_s in enumerate(self.nodes):
            nodes_to_connect = list(self.nodes)
            nodes_to_connect.pop(i)
            nodes_to_connect = nodes_to_connect[i:]
            for node_r in nodes_to_connect:
                arrow = "<->"
                # make arrows longer between tinyFaaS and cloud nodes
                if ("Node" in node_r) ^ ("Node" in node_s):
                    arrow = "<-->"
                uml_code.append(f"{node_s} {arrow} {node_r}")

        uml_code.append("@enduml")  # end uml code

        with open(os.path.abspath("diagram_uml.puml"), "w+") as f:
            for line in uml_code:
                f.write(line + "\n")