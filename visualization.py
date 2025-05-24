# Visualize optimization results
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import numpy as np

from creation import Problem
from optimization import Optimization
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


def abbreviate_wf_names_aggressive(workflows: list[str]):
    new_names = []
    for workflow in workflows:
        # split each name by '_'
        sections = workflow.split('_')
        # take first letter from each section
        new_name = ''.join([word[0] for word in sections])
        new_names.append(new_name)
    return new_names


def abbreviate_node_names(nodes: list):
    n = [name.replace("Node-", "Nd") for name in nodes]
    return [name.replace("Google", "Ggl") for name in n]


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

        # cast all numerics to integer
        col_types = {header: 'int32' for header in self.table.columns[1:]}
        self.table = self.table.astype(col_types)

    def insert_batch_vars(self, vs: dict):
        x, y = vs['x'][0], vs['y'][0]
        entries = [f"{int(x[i])}x{int(y[i])}" for i in range(len(x))]
        self.table.loc[len(x)] = ['X*Y'] + entries

    def print(self):
        print("_" * len(self.name))
        # print(self.name)
        pd.set_option('display.max_colwidth', None)
        print(self.table)


# Visualize and annotate optimization data using Pandas
class WorkflowsTable:
    def __init__(self, wf: Workflows, cld: CloudsInfo, data: dict):

        # list of workflows to print as tables
        self.workflow_tables = []

        P = data['P']
        self.flagBatch = False
        if 'x' in data:
            self.flagBatch = True
            x = data['x']
            y = data['y']

        # annotate node names
        num_nodes = P.shape[2]  # number of nodes (0: wf x 1: fun x 3: nodes)
        names_cloud = list(cld.names)
        num_tiny = num_nodes - len(names_cloud)
        nodes = [f"Nd-{i}" for i in range(num_tiny)]
        [nodes.append(cloud_name) for cloud_name in names_cloud]

        # annotate function names and create tables
        for index, workflow in enumerate(wf.names_workflows):
            name = abbreviate_wf_name(workflow)
            names_funs = wf.names_funs[index]
            table = WorkflowTable(name=name, functions=names_funs, nodes=nodes, p_matrix=P[index])
            if self.flagBatch:
                x = data['x'][index]
                y = data['y'][index]
                table.insert_batch_vars({'x': x, 'y': y})
            self.workflow_tables.append(table)

    def print(self):
        # list to save tables as CSV
        csv = []
        for table in self.workflow_tables:
            table.print()
            # breakup csv string into list elements, reject last empty line
            csv = csv + table.table.to_csv().split('\r\n')[:-1]
            # save as image using df2img (lookup in 'Python Packages' tab)

        # write CSV tables to file
        with open(os.path.abspath(f"{SAVE_PATH}optimization_result.csv"), "w+") as f:
            for line in csv:
                f.write(line + "\n")


# Superclass for creating UML-diagrams
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
                # Check if function properties are zero
                if sum([wf.funs_times[i, j], wf.funs_data[i, j], wf.funs_sizes[i, j]]) == 0:
                    self.uml_code.append(f"() \"{function}\" as {wf_name_abbrev}_{function}")
                else:
                    self.uml_code.append(f"json \"{function}\" as {wf_name_abbrev}_{function} {{")
                    self.uml_code.append(f"\"time\": {wf.funs_times[i, j]},")
                    self.uml_code.append(f"\"data\": {wf.funs_data[i, j]},")
                    if wf.funs_counts[i, j] > 1:
                        self.uml_code.append(f"\"size\": {wf.funs_sizes[i, j]},")
                        self.uml_code.append(f"\"count\": {wf.funs_counts[i, j]}")
                    else:
                        self.uml_code.append(f"\"size\": {wf.funs_sizes[i, j]}")
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


class NetworkUML(DiagramUML):
    def __init__(self, net: LocalNetwork, cld: CloudsInfo):
        # initialize uml-diagram superclass
        super().__init__(title="Network Properties", file_name="network_uml")
        self.uml_code.append("left to right direction")

        # cloud nodes with pricing displayed
        for node in range(cld.num):
            self.uml_code.append(f'cloud {cld.names[node]} [')  # start cloud
            self.uml_code.append(f'\t<b>{cld.names[node]}')
            self.uml_code.append('\tCosts')
            self.uml_code.append(f'\tRun (GBs): {cld.prices_ram[node]}')
            self.uml_code.append(f'\tTransfer (GB): {cld.prices_transfer_up[node]}')
            self.uml_code.append(f'\tRequest: {cld.prices_start[node]}')
            self.uml_code.append(f'\tp_factor: {cld.p_factors[node]}')
            self.uml_code.append(']')  # close cloud
        self.uml_code.append('\n')

        # Frame local for enclosing tinyFaaS nodes and router
        self.uml_code.append('frame Local{')  # start package
        self.uml_code.append('\tport router')
        nodes_codes = []
        for node in range(net.num):
            node_code = f'[Node-{node}\\n' \
                        f'RAM: {net.rams[node]}GB\\n' \
                        f'p_factor: {net.p_factors[node]}]'
            nodes_codes.append(node_code)
            self.uml_code.append(f'\t{node_code}')

        self.uml_code.append(f'\tnote as N\n\t\tTotal RAM: {sum(net.rams)}GB\n\tend note')
        self.uml_code.append('}\n')  # close package

        # connect nodes and router with latency annotated
        for code in nodes_codes:
            self.uml_code.append(f'{code} <-> router : {net.latency_local}')

        # connect cloud to router
        for node in range(cld.num):
            self.uml_code.append(f'{cld.names[node]} <--> router : {cld.latency[node]}')

    def code_diagram(self):
        self.write_uml_file()


class DeploymentUML(DiagramUML):
    def __init__(self, wf: Workflows, net: LocalNetwork, cld: CloudsInfo, data: dict):
        # initialize uml-diagram superclass
        super().__init__(title="Optimal Deployment Solution", file_name="deployment_uml")

        # Create dictionary to hold information about occupiers of each node
        self.nodes = {f"Node_{i}": {} for i in range(net.num)}
        self.nodes.update({cloud_name: {} for cloud_name in cld.names})
        self.num_tiny = net.num

        P = data['P']
        self.flagBatch = False
        if 'x' in data:
            self.flagBatch = True
            x = data['x']
            y = data['y']

        for i_node, node in enumerate(self.nodes):
            # Assign node ram limit
            if i_node < net.num:
                self.nodes[node]["ram_limit"] = net.rams[i_node]
            else:
                self.nodes[node]["ram_limit"] = np.inf

            self.nodes[node]["ram_used"] = 0
            self.nodes[node]["workflows"] = {}
            for i_wf, workflow in enumerate(wf.names_workflows):
                # Check if any function within current workflow is deployed to this node
                if sum(abs(P[i_wf, :, i_node])) > 0:
                    self.nodes[node]["workflows"][workflow] = []
                    for i_f, function in enumerate(wf.names_funs[i_wf]):
                        flagXY = False
                        if self.flagBatch and wf.funs_counts[i_wf, i_f] > 1:
                            flagXY = True
                            ram_user = x[i_wf, :, i_node].item()
                        else:
                            ram_user = P[i_wf, i_f, i_node]
                        # Assign used ram
                        self.nodes[node]["ram_used"] += ram_user * (wf.funs_data[i_wf, i_f]
                                                                    + wf.funs_sizes[i_wf, i_f])
                        # Assign deployed function(s)
                        if abs(P[i_wf, i_f, i_node]) > 0:
                            self.nodes[node]["workflows"][workflow].append(f"{function}: "
                                                                           f"{round(P[i_wf, i_f, i_node])}"
                                                                           f"/{wf.funs_counts[i_wf, i_f]}")
                            if flagXY:
                                self.nodes[node]["workflows"][workflow].append(f"\t\t ({int(x[i_wf, :, i_node])}"
                                                                               f"x{int(y[i_wf, :, i_node])})")

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


class PlotObjectives:
    def __init__(self, op: Optimization, wf: Workflows, pb: Problem):
        time_divisor = 1
        ratio_time_cost = (op.obj_latency.getValue() + op.obj_time.getValue()) \
                          / (op.obj_transfer.getValue() + op.obj_ram.getValue())
        if ratio_time_cost > 1000:
            time_divisor = 1000

        # format to 3 decimal points or scientific if zero
        format_float = lambda x: f"{x:.3e}" if x != 0 and f"{x:.3f}" == "0.000" else f"{x:.3f}"

        self.total_objective = op.obj.getValue()
        self.main_objectives = {"latency": op.obj_latency.getValue(),
                                "time/" + str(time_divisor): op.obj_time.getValue() / time_divisor,
                                "startup_time": op.obj_startup_time.getValue(),
                                "transfer": op.obj_transfer.getValue(),
                                "ram": op.obj_ram.getValue(),
                                "startup_cost": op.obj_startup_cost.getValue(),
                                "control": op.obj_control.getValue()}

        # print some data
        print()
        print(f'Objective: {self.total_objective:.3f}')
        objs = self.main_objectives
        [print(f'{key}: {format_float(value)}') for key, value in objs.items()]
        [print(f"{item[0]}: {item[1]}") for item in op.report.items()]

        # Get workflow objective costs
        self.workflows_names = [abbreviate_wf_name(workflow) for workflow in wf.names_workflows]
        self.workflows_latency = []
        self.workflows_time = []
        self.workflows_transfer = []
        self.workflows_ram = []
        self.workflows_control = []  # The online control time
        for workflow in range(wf.num_workflows):
            workflow_latency = 0
            workflow_time = 0
            workflow_transfer = 0
            workflow_ram = 0
            workflow_control = 0
            workflow_control_flag = True
            for function in range(len(wf.names_funs[workflow]) - 1):
                function_latency = []
                function_time = []
                for nd_send in range(pb.num_nodes):
                    for nd_recv in range(pb.num_nodes):
                        function_latency.append(op.obj_latency_details[workflow, function, nd_send, nd_recv].getValue())
                        workflow_transfer += op.obj_transfer_details[workflow, function, nd_send, nd_recv].getValue()
                        if workflow_control_flag:
                            workflow_control_flag = False
                            workflow_control += op.T_f[workflow, nd_send, nd_recv].X
                    function_time.append(op.obj_time_details[workflow, function, nd_send].getValue())
                    workflow_ram += op.obj_ram_details[workflow, function, nd_send].getValue()
                workflow_latency += max(function_latency)
                workflow_time += max(function_time)
            self.workflows_latency.append(workflow_latency)
            self.workflows_time.append(workflow_time)
            self.workflows_transfer.append(workflow_transfer)
            self.workflows_ram.append(workflow_ram)
            self.workflows_control.append(workflow_control)

        # Pie Chart of totals
        plt.pie(x=list(self.main_objectives.values()), labels=list(self.main_objectives.keys()),
                wedgeprops=dict(width=0.5))
        plt.title("Objective Costs")
        plt.text(x=-1, y=-1.2, s=f"Total: {self.total_objective:.3f}, Weights: {op.w_1:.3f}, {op.w_2:.3f}")
        plt.savefig(f"{SAVE_PATH}objectives_pie.png")

        workflows_names = self.workflows_names
        if np.mean([len(wf_name) for wf_name in self.workflows_names]) > 7:
            workflows_names = abbreviate_wf_names_aggressive(workflows=self.workflows_names)

        # subplot of the 4 quad objectives
        self.fig, self.ax = plt.subplots(3, 2)
        self.ax[0, 0].bar(x=workflows_names, height=self.workflows_latency)
        self.ax[0, 0].set_title("Workflows' Total Latency [s]")
        self.ax[0, 1].bar(x=workflows_names, height=self.workflows_time)
        self.ax[0, 1].set_title("Workflows' Total Time [s]")
        self.ax[1, 0].bar(x=workflows_names, height=self.workflows_transfer)
        self.ax[1, 0].set_title("Workflows' Data Transfer Cost [$]")
        self.ax[1, 1].bar(x=workflows_names, height=self.workflows_ram)
        self.ax[1, 1].set_title("Workflows' Running Cost [$]")
        self.ax[2, 0].bar(x=workflows_names, height=self.workflows_control)
        self.ax[2, 0].set_title("Workflows' Online Control Latency [s]")
        self.ax[2, 1].bar(x=abbreviate_node_names(pb.name_nodes), height=[op.T_s[i].X for i in op.T_s])
        self.ax[2, 1].set_title("Node additional startup time [s]")
        self.fig.tight_layout()
        self.fig.savefig(f"{SAVE_PATH}wf_objectives.eps")

        # Save the figure object to a pickle file
        with open('wf_objectives.pkl', 'wb') as file:
            pkl.dump((self.fig, self.ax), file)

        # Save optimization variables to a pickle file
        sol_vars = {var.VarName: var.X for var in op.model.getVars()}
        with open('solution.pkl', 'rb') as file:
            pkl.dump(sol_vars, file)

    def show_plots(self):
        self.fig.show()
        plt.show()
