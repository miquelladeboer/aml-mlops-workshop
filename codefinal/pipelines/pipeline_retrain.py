# Required Libraries
from azureml.core import Workspace, Experiment, Datastore
from azureml.core.runconfig import MpiConfiguration, RunConfiguration
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.dataset import Dataset

from azureml.core.conda_dependencies import CondaDependencies

from azureml.train.dnn import PyTorch
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig, PrimaryMetricGoal)

from azureml.pipeline.steps import HyperDriveStep, PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData

import os



workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Retrieve datastore/datasets
# retrieve datastore
datastore_name = 'workspaceblobstore'
datastore = Datastore.get(workspace, datastore_name)

# retrieve subset datasets used for training
subset_dataset_train = Dataset.get_by_name(workspace,
                                           name='newsgroups_subset_train')
subset_dataset_test = Dataset.get_by_name(workspace,
                                          name='newsgroups_subset_test')

# retrieve full datasets used for training full model
# retrieve subset datasets used for training
dataset_train = Dataset.get_by_name(workspace,
                                    name='newsgroups_train')
dataset_test = Dataset.get_by_name(workspace,
                                   name='newsgroups_test')

# denife output dataset for run metrics JSON file
metrics_output_name = 'metrics_output'
metrics_data = PipelineData(name='metrics_data',
                            datastore=datastore,
                            pipeline_output_name=metrics_output_name)

# Define the compute target
compute_target_hyper = workspace.compute_targets["hypercomputegpu"]
compute_target_fullmodel = workspace.compute_targets["fullcomputegpu"]

# Define Run Configuration
estimator = PyTorch(
    entry_script='train.py',
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'modeling'
    ),
    compute_target=compute_target_hyper,
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
    ],
    inputs=[
        subset_dataset_train.as_named_input('subset_train'),
        subset_dataset_train.as_named_input('subset_test')
    ]
)

# Set parameters for search
param_sampling = BayesianParameterSampling({
    "learning_rate": uniform(0.05, 0.1),
    "num_epochs": choice(5, 10, 15),
    "batch_size": choice(150, 200),
    "hidden_size": choice(50, 100)
})

# Define the pipeline step
hypertuning = HyperDriveStep(
            name='hypertrain',
            hyperdrive_config=HyperDriveConfig(
                estimator=estimator,
                hyperparameter_sampling=param_sampling,
                policy=None,
                primary_metric_name="accuracy",
                primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                max_total_runs=2,
                max_concurrent_runs=None
            ),
            estimator_entry_script_arguments=[],
            inputs=[
                    subset_dataset_train.as_named_input('subset_train'),
                    subset_dataset_test.as_named_input('subset_test')
                    ],
            outputs=[],
            metrics_output=metrics_data,
            allow_reuse=True,
            version=None
)

# Define the conda dependencies
cd = CondaDependencies(
    conda_dependencies_file_path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../',
        'conda_dependencies.yml'
    )
)

# Runconfig
amlcompute_run_config = RunConfiguration(conda_dependencies=cd)
amlcompute_run_config.environment.docker.base_image = "pytorch/pytorch"
amlcompute_run_config.environment.spark.precache_packages = False

fullmodel = PythonScriptStep(
    script_name="train.py",
    arguments=[],
    inputs=[
            subset_dataset_train.as_named_input('subset_train'),
            subset_dataset_test.as_named_input('subset_test'),
            metrics_data
    ],
    outputs=[],
    compute_target=compute_target_fullmodel,
    runconfig=amlcompute_run_config,
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'modeling'
    )
)

# Attach step to the pipelines
pipeline = Pipeline(workspace=workspace, steps=[hypertuning, fullmodel])

# Submit the pipeline
# Define the experiment
experiment = Experiment(workspace, 're-train')

# Run the experiment
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion()
