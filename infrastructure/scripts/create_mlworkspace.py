"""
Scripts for the createin of an ML workspace. It is recommended to deploy
your workspaced through Azure DevOps using the ARM templates, but for quick
deployment, you vcan you this script.
-----------------------------
- deploy Azure ML workspace
- deploy dafault strogae account
- deploy Azure Key Valut
- Deploy Application Insights
"""
import os
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication

# Create the workspace using the specified parameters
ws = Workspace.create(
    name='<Name of workspace>',
    subscription_id='<ID>',
    resource_group='<Name of recourse group>',
    location='<location>',
    create_resource_group=True,
    sku='basic',
    exist_ok=True,
    auth=AzureCliAuthentication()
)

print(ws.get_details())

# write the details of the workspace to a configuration file in the project root
ws.write_config(
    path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../'
    )
)