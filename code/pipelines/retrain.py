from azureml.core import Workspace, Experiment, Datastore
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.dataset import Dataset
from azureml.train.estimator import Estimator
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig, PrimaryMetricGoal)
from azureml.pipeline.steps import HyperDriveStep, PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core.runconfig import RunConfiguration

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
# retrieve subset datasets used for trainingK
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
compute_target_hyper = workspace.compute_targets["slowcluster"]
compute_target_fullmodel = workspace.compute_targets["fastcluster"]

# Load run Config
run_config = RunConfiguration.load(
    path=os.path.join(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        "environments/PyTorch/RunConfig/runconfig_pytorch.yml",
        )),
    name="pytorch"
)

# Set parameters for search
param_sampling = BayesianParameterSampling({
    "learning_rate": uniform(10e-6, 1e0),
    "num_epochs": choice(50, 100, 150),
    "batch_size": choice(100, 300),
    "hidden_size": choice(10000, 50000)
})

# Define Run Configuration
estimator = Estimator(
    entry_script='train.py',
    script_params={
        '--dataset': 'subset_',
        '--models': 'deepelaerning',
        '--data_local': False,
        '--fullmodel': False,
    },
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'modeling'
    ),
    compute_target=compute_target_hyper,
    run_config=run_config,
    inputs=[
        subset_dataset_train.as_named_input('subset_train'),
        subset_dataset_train.as_named_input('subset_test')
    ]
)

# Define the pipeline step
hypertuning = HyperDriveStep(
            name='hypertrain',
            hyperdrive_config=HyperDriveConfig(
                estimator=estimator,
                hyperparameter_sampling=param_sampling,
                policy=None,
                primary_metric_name="accuracy",
                primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                max_total_runs=80,
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


fullmodel = PythonScriptStep(
    name="fullmodel",
    script_name="train.py",
    arguments=['--dataset', '',
               '--models', 'deeplearning',
               '--data_local', False,
               '--fullmodel', True],
    inputs=[
            dataset_train.as_named_input('train'),
            dataset_test.as_named_input('test'),
            metrics_data
    ],
    outputs=[],
    compute_target=compute_target_fullmodel,
    run_config=run_config,
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
experiment = Experiment(workspace, 'pipeline re-train')

# Run the experiment
pipeline_run = experiment.submit(pipeline)
