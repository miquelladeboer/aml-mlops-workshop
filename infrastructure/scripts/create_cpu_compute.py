from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Create compute target if not present
# Choose a name for your CPU cluster
cpu_cluster_name = "hypercomputecpu"

# Verify that cluster does not exist already
try:
    cu_cluster = ComputeTarget(workspace=workspace, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D3_V2',
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(workspace, cpu_cluster_name,
                                       compute_config)

cpu_cluster.wait_for_completion(show_output=True)
