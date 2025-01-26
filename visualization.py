# Visualize optimization results
import pandas as pd
from dataclasses import dataclass, field
import numpy as np

from parsing import *

SAVE_PATH = "report//"  # relative save path


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
        workflow_name = workflow_name.replace("workflow", "wf")
    if "test" in workflow_name:
        workflow_name = workflow_name.replace("test", "tst")
    if "tiny" in workflow_name:
        workflow_name = workflow_name.replace("tiny", "tny")
    if "-" in workflow_name:
        workflow_name = workflow_name.replace("-", "_")
    return workflow_name


# Annotate workflow deployment table using pandas
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
        self.table.index.name = self.name

    def print(self):
        print("_" * len(self.name))
        # print(self.name)
        pd.set_option('display.max_colwidth', None)
        print(self.table)


# Visualize and annotate optimization result using Pandas
class WorkflowsTable:
    def __init__(self, wf: Workflows, cld: CloudsInfo, P: np.ndarray):
        # list of workflows to print as tables
        self.workflow_tables = []

        # annotate node names
        num_nodes = P.shape[2]  # number of nodes (0: wf x 1: fun x 3: nodes)
        names_cloud = list(cld.names_cloud)
        num_tiny = num_nodes - len(names_cloud)
        nodes = [f"Nd-{i}" for i in range(num_tiny)]
        [nodes.append(cloud_name) for cloud_name in names_cloud]

        # annotate function names and create tables
        for index, workflow in enumerate(wf.names_workflows):
            name = workflow
            names_funs = wf.names_funs[index]
            self.workflow_tables.append(WorkflowTable(name=name, functions=names_funs, nodes=nodes, p_matrix=P[index]))

    def print(self):
        # list to save tables as CSV
        csv = []
        for table in self.workflow_tables:
            table.print()
            csv = csv + table.table.to_csv().split('\r\n')  # breakup single string into list elements

        # write CSV tables to file
        with open(os.path.abspath(f"{SAVE_PATH}optimization_result.csv"), "w+") as f:
            for line in csv:
                f.write(line + "\n")



class DiagramUML:
    def __init__(self, title: str, file_name: str):
        # Creates lines of UML code using a list.
        # Each element in the list is a separate line in the UML file
        self.uml_code = ["@startuml", "\n"]  # first line in a 'uml' file
        # diagram title
        self.uml_code.append(f"title {title}")
        # diagram save path
        self.save_path = f"{SAVE_PATH}{file_name}.puml"

    def write_uml_file(self):
        self.uml_code.append("@enduml")  # end uml code

        with open(os.path.abspath(self.save_path), "w+") as f:
            for line in self.uml_code:
                f.write(line + "\n")


class WorkflowsUML(DiagramUML):
    def __init__(self, wf: Workflows):
        # initialize uml-diagram superclass
        super().__init__(title="Workflows Properties", file_name="workflows_uml")

        # Diagram flow from left to right
        self.uml_code = self.uml_code + ["left to right direction", "\n"]

        for i, workflow in enumerate(wf.names_workflows):
            # abbreviate workflow name for usability
            wf_name_abbrev = abbreviate_wf_name(workflow)

            # label series of functions with workflow name
            self.uml_code.append(f"package \"{workflow}\"{{")
            # Json objects to encapsulate functions' name and properties
            for j, function in enumerate(wf.names_funs[i]):
                # Check if function is local (properties are zero)
                if j in wf.funs_local[i]:  # sketchy! assumes every workflow has at least one local function
                    self.uml_code.append(f"() \"{function}\" as {wf_name_abbrev}_{function}")
                else:
                    self.uml_code.append(f"json \"{function}\" as {wf_name_abbrev}_{function} {{")
                    self.uml_code.append(f"\"time\": {wf.funs_times[i][j]},")
                    self.uml_code.append(f"\"data\": {wf.funs_data[i][j]},")
                    self.uml_code.append(f"\"size\": {wf.funs_sizes[i][j]}")
                    self.uml_code.append("}")

            # Connect consecutive functions with arrows
            for j in range(len(wf.names_funs[i]) - 1):
                self.uml_code.append(f"{wf_name_abbrev}_{wf.names_funs[i][j]} -->"
                                     f" {wf_name_abbrev}_{wf.names_funs[i][j + 1]}")

            # close workflow package
            self.uml_code.append("}")

            self.uml_code.append("\n")

    def code_diagram(self):
        self.write_uml_file()


class DeploymentUML(DiagramUML):
    def __init__(self, wf: Workflows, net: LocalNetwork, cld: CloudsInfo, P: np.ndarray):
        # initialize uml-diagram superclass
        super().__init__(title="Optimal Deployment Solution", file_name="nodes_assigned_uml")

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
                        self.nodes[node]["ram_used"] += P[i_wf, i_f, i_node] * wf.funs_data[i_wf][i_f]
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
        node_code.append(f"RAM usage: {round(node_data['ram_used'], 3)}/{node_data['ram_limit']}")
        return node_code

    def code_diagram(self):
        for i, node in enumerate(self.nodes):
            # declare a deployment diagram entity based on node type
            if i < self.num_tiny:
                self.uml_code.append(f"node {node}[")
            else:
                self.uml_code.append(f"cloud {node}[")

            # fill in node information coded with indentation
            for line in self.code_a_node(node):
                self.uml_code.append("  " + line)

            self.uml_code.append("]")  # close node description

        self.uml_code.append("\n")  # empty line after node description

        # connect nodes with arrows
        for i, node_s in enumerate(self.nodes):
            # list of nodes to connect to node_s
            nodes_to_connect = list(self.nodes)
            # remove node_s from list to not connect it to itself
            nodes_to_connect.pop(i)
            # list should have nodes not connected beforehand to node_s
            nodes_to_connect = nodes_to_connect[i:]
            # loop over nodes in list to connect to node_s
            for node_r in nodes_to_connect:
                arrow = "<->"
                # make arrows longer between tinyFaaS and cloud nodes
                if ("Node" in node_r) ^ ("Node" in node_s):
                    arrow = "<-->"
                self.uml_code.append(f"{node_s} {arrow} {node_r}")

        self.write_uml_file()
