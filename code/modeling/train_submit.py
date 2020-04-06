"""
Training submitter

Facilitates (remote) training execution through the Azure ML service.
"""
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.train.estimator import Estimator
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.dataset import Dataset
from azureml.core.runconfig import RunConfiguration
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig, PrimaryMetricGoal)


# Define comfigs
dataset = ''
models = 'randomforest'
data_local = False

# define compute
compute_target = 'alwaysoncluster'

# If deep learning define hyperparameters
# Set parameters for search
param_sampling = BayesianParameterSampling({
    "learning_rate": uniform(0.05, 0.1),
    "num_epochs": choice(5, 10, 15),
    "batch_size": choice(150, 200),
    "hidden_size": choice(50, 100)
})

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# define script parameters
script_params = [
    '--dataset', "subset_",
    '--models', models,
    '--data_local', data_local,
]

# get data stores if from azure
if data_local is False:
    dataset_train = Dataset.get_by_name(workspace,
                                        name='newsgroups_' + dataset + 'train')
    dataset_test = Dataset.get_by_name(workspace,
                                       name='newsgroups_' + dataset + 'test')

if models != 'deeplearning':
    if data_local is True:
        # Define Run Configuration
        est = Estimator(
            entry_script='train.py',
            script_params=script_params,
            source_directory=os.path.dirname(os.path.realpath(__file__)),
            compute_target=compute_target,
            user_managed=True,
            use_docker=False,
        )

    if data_local is False:
        # Load run Config
        run_config = RunConfiguration.load(
            path=os.path.join(os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../..",
                "environments/sklearn/RunConfig/runconfig_sklearn.yml",
                )),
            name="sklearn"
        )
   
        est = ScriptRunConfig(
            script='train.py',
            source_directory=os.path.dirname(os.path.realpath(__file__)),
            run_config=run_config,
            arguments=script_params
        )

    # Define the ML experiment
    experiment = Experiment(workspace, "explore_sklearn_models")
    # Submit experiment run, if compute is idle, this may take some time')
    run = experiment.submit(est)

if models == 'deeplearning':
    # Load run Config
    run_config = RunConfiguration.load(
        path=os.path.join(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../..",
            "environments/PyTorch/RunConfig/runconfig_pytorch.yml",
            )),
        name="pytorch"
    )

    # Define Run Configuration
    estimator = Estimator(
        entry_script='train.py',
        script_params=script_params,
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        compute_target=workspace.compute_targets[compute_target],
        run_config=run_config,
        inputs=[
            dataset_train.as_named_input(dataset + 'train'),
            dataset_train.as_named_input(dataset + 'test')
        ]
    )

    # Define multi-run configuration
    hyperdrive_run_config = HyperDriveConfig(
        estimator=estimator,
        hyperparameter_sampling=param_sampling,
        policy=None,
        primary_metric_name="accuracy",
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=10,
        max_concurrent_runs=None
    )

    # Define the ML experiment
    experiment = Experiment(workspace, "newsgroups_train")

    # Submit the experiment
    run = experiment.submit(hyperdrive_run_config)

run.wait_for_completion()
