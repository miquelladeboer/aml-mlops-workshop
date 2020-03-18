from azureml.train.hyperdrive import HyperDriveRun
from azureml.core import Workspace, Experiment
from azureml.core.runconfig import MpiConfiguration
import os
from azureml.core.authentication import AzureCliAuthentication
from azureml.train.dnn import PyTorch
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Create compute target if not present
# Choose a name for your CPU cluster
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


# Define the ML experiment
experiment = Experiment(workspace, 'newsgroups_train_hypertune_gpu')

# Get all the runs in the experiment
generator = experiment.get_runs(type=None,
                                tags=None,
                                properties=None,
                                include_children=False)
run = next(generator)
# Select the last run
parent = HyperDriveRun(experiment, run_id=run.id)

# Select the best run from all submitted
best_run = parent.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()

# Best set of parameters found
parameter_values = best_run.get_details()['runDefinition']['arguments']
best_parameters = dict(zip(parameter_values[::2], parameter_values[1::2]))
best_model_parameters = best_parameters.copy()


# Define a final training run with model's best parameters
model_est = PyTorch(
    entry_script='traindeep.py',
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    script_params=best_model_parameters,
    compute_target=workspace.compute_targets[gpu_cluster_name],
    distributed_training=MpiConfiguration(),
    framework_version='1.4',
    use_gpu=True,
    pip_packages=[
        'numpy==1.15.4',
        'pandas==0.23.4',
        'scikit-learn==0.20.1',
        'scipy==1.0.0',
        'matplotlib==3.0.2',
        'utils==0.9.0',
    ]
)

# Define the ML experiment
experiment = Experiment(workspace, "newsgroups_train_fullmodel")

# Submit the experiment
model_run = experiment.submit(model_est)

model_run_status = model_run.wait_for_completion(wait_post_processing=True)
