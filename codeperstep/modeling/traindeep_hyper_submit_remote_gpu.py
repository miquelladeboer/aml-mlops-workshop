from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig, PrimaryMetricGoal)
from azureml.core import Workspace, Experiment
from azureml.core.runconfig import MpiConfiguration
import os
from azureml.train.hyperdrive.parameter_expressions import uniform, choice


from azureml.core.authentication import AzureCliAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.dnn import PyTorch

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Create compute target if not present
# Choose a name for your CPU cluster
gpu_cluster_name = "hypercomputegpu"

# Verify that cluster does not exist already
try:
    gpu_cluster = ComputeTarget(workspace=workspace, name=gpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC6',
                                                           max_nodes=4)
    gpu_cluster = ComputeTarget.create(workspace, gpu_cluster_name,
                                       compute_config)

gpu_cluster.wait_for_completion(show_output=True)


# Define Run Configuration
estimator = PyTorch(
    entry_script='traindeep.py',
    source_directory=os.path.dirname(os.path.realpath(__file__)),
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
        'utils==0.9.0'
    ]
)

# Set parameters for search
param_sampling = BayesianParameterSampling({
    "learning_rate": uniform(0.05, 0.1),
    "num_epochs": choice(5, 10, 15),
    "batch_size": choice(150, 200),
    "hidden_size": choice(50, 100)
})

# Define multi-run configuration
hyperdrive_run_config = HyperDriveConfig(
    estimator=estimator,
    hyperparameter_sampling=param_sampling,
    policy=None,
    primary_metric_name="accuracy",
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=80,
    max_concurrent_runs=None
)

# Define the ML experiment
experiment = Experiment(workspace, "newsgroups_train_hypertune_gpu")

# Submit the experiment
hyperdrive_run = experiment.submit(hyperdrive_run_config)
hyperdrive_run.wait_for_completion()

