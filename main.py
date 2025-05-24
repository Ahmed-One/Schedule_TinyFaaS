# Parse the data,create the problem, run the optimization and visualize the results
from optimization import *
from visualization import WorkflowsTable, DeploymentUML, WorkflowsUML, NetworkUML, PlotObjectives


def run():
    # Parse the JSON data
    workflows = Workflows(datapath="requirements//workflows_DOMPC.json")
    network = LocalNetwork(datapath="requirements//network.json")
    cloud = CloudsInfo(datapath="requirements//cloud.json")

    # Check if enough tinyFaaS nnodes are available for the amount of workflows
    check_network_validity(num_workflows=workflows.num_workflows, num_tiny=network.num)

    # report dictionary for recording solve statistics
    report = {}

    # Create problem parameters and optimize for the solution
    problem = Problem(wf=workflows, net=network, cld=cloud)
    op = Optimizer6(wf=workflows, net=network, pb=problem)
    op.normalize_weights_simple(wf=workflows, pb=problem)
    # op.w_1 = 1
    # op.w_2 = 1  # 470000
    op.setup()
    op.solve()
    assert op.model.Status != GRB.INFEASIBLE, "Model is Infeasible!"

    # Print data
    results = {'P': op.get_result(v='P')}
    if op.flagBatch:
        results['x'] = op.get_result(v='x')
        results['y'] = op.get_result(v='y')

    tables = WorkflowsTable(wf=workflows, cld=cloud, data=results)
    tables.print()

    workflows_diagram = WorkflowsUML(wf=workflows)
    workflows_diagram.code_diagram()

    network_diagram = NetworkUML(net=network, cld=cloud)
    network_diagram.code_diagram()

    assigned_nodes_diagram = DeploymentUML(wf=workflows, net=network, cld=cloud, data=results)
    assigned_nodes_diagram.code_diagram()

    objectives_plot = PlotObjectives(op=op, wf=workflows, pb=problem)
    objectives_plot.show_plots()

    pass


if __name__ == '__main__':
    run()
