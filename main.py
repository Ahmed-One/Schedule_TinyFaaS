# Parse the data,create the problem, run the optimization and visualize the results
from optimization import *
from visualization import WorkflowsTable, DeploymentUML, WorkflowsUML, NetworkUML


def run():
    # Parse the JSON data
    workflows = Workflows(datapath="requirements//workflows.json")
    network = LocalNetwork(datapath="requirements//network.json")
    cloud = CloudsInfo(datapath="requirements//cloud.json")

    # Check if enough tinyFaaS nnodes are available for the amount of workflows
    check_network_validity(num_workflows=workflows.num_workflows, num_tiny=network.num_tiny)

    # Create problem parameters and optimize the for the solution
    problem = Problem(wf=workflows, net=network, cld=cloud)
    op = Optimizer(wf=workflows, net=network, pb=problem)
    op.w_1 = 1
    op.w_2 = 470000
    op.solve()

    # Print result
    P_out = np.reshape(np.array([item.x for item in op.model.getVars()]),
                       (workflows.num_workflows, workflows.num_funs, problem.num_nodes))
    tables = WorkflowsTable(wf=workflows, cld=cloud, P=P_out)
    tables.print()

    workflows_diagram = WorkflowsUML(wf=workflows)
    workflows_diagram.code_diagram()

    network_diagram = NetworkUML(net=network, cld=cloud)
    network_diagram.code_diagram()

    assigned_nodes_diagram = DeploymentUML(wf=workflows, net=network, cld=cloud, P=P_out)
    assigned_nodes_diagram.code_diagram()

    pass


if __name__ == '__main__':
    run()
