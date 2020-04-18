"""
Helper to get run details for debugging
"""
import os
from azureml.core import Workspace, Experiment, Run
from azureml.core.authentication import AzureCliAuthentication


# load workspace
ws = Workspace.from_config(
    auth=AzureCliAuthentication(),
    path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'config.json'
    )
)
experiment = Experiment(
    workspace=ws,
    name="multitask_ssd"
)

run = Run(
    experiment,
    run_id="a9858223-b3b7-4272-98a4-33bfee9f67e9"
)

print(run.get_details())

# print(run.get_metrics())