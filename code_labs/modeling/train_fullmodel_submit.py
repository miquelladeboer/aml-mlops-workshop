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
    entry_script='train.py',
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    script_params=best_model_parameters,
    compute_target=workspace.compute_targets[gpu_cluster_name],
    distributed_training=MpiConfiguration(),
    framework_version='1.4',
    use_gpu=True,
    conda_dependencies_file=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../',
            'conda_dependencies.yml'
        )
)

# Define the ML experiment
experiment = Experiment(workspace, "newsgroups_train_fullmodel")

# Submit the experiment
model_run = experiment.submit(model_est)

model_run_status = model_run.wait_for_completion(wait_post_processing=True)
