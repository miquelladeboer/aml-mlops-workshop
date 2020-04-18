"""
Helper to get the workspace id for debugging purposes
"""
import os
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication


# load workspace
ws = Workspace.from_config(
    auth=AzureCliAuthentication(),
    path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'config.json'
    )
)

print(ws._workspace_id)
