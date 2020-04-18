"""
Helper to get run details for debugging purposes
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

for ds in ws.datastores:
    print(ds)
