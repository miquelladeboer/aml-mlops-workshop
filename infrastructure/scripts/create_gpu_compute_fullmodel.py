from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Create compute target if not present
# Choose a name for your GPU cluster
gpu_cluster_name = "fullcomputegpu"

# Verify that cluster does not exist already
try:
    gpu_cluster = ComputeTarget(workspace=workspace, name=gpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC12',
                                                           max_nodes=8)
    gpu_cluster = ComputeTarget.create(workspace, gpu_cluster_name,
                                       compute_config)

gpu_cluster.wait_for_completion(show_output=True)