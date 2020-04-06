import os
from azureml.core import Workspace, Datastore
from azureml.core.authentication import AzureCliAuthentication

OUTPUTSFOLDER = "outputs/raw_data"

datastore_name = 'workspaceblobstore'

# get existing ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# retrieve an existing datastore in the workspace by name
datastore = Datastore.get(workspace, datastore_name)

# upload files
datastore.upload(
    src_dir=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../',
        OUTPUTSFOLDER
    ),
    target_path=None,
    overwrite=True,
    show_progress=True
)
