from azureml.core import Workspace, Experiment, Datastore
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.dataset import Dataset
from azureml.train.dnn import PyTorch
from azureml.core.runconfig import MpiConfiguration
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig, PrimaryMetricGoal)
from azureml.pipeline.steps import HyperDriveStep, PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core.runconfig import RunConfiguration

import os

# Set parameters for search
param_sampling = BayesianParameterSampling({
    "learning_rate": uniform(10e-6, 1e0),
    "num_epochs": choice(10, 20),
    "batch_size": choice(10, 20, 50, 100, 200, 300, 500, 1000),
    "hidden_size": choice(300, 400)
})

workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Retrieve datastore/datasets
# retrieve datastore
datastore_name = 'workspaceblobstore'
datastore = Datastore.get(workspace, datastore_name)

# define data set names
input_name_train_sub = 'newsgroups_subset_train'
input_name_test_sub = 'newsgroups_subset_test'

input_name_train = 'newsgroups_train'
input_name_test = 'newsgroups_test'

# retrieve subset datasets used for training
dataset_train_sub = Dataset.get_by_name(workspace, name=input_name_train_sub)
dataset_test_sub = Dataset.get_by_name(workspace, name=input_name_test_sub)

# retrieve full datasets used for training full model
# retrieve subset datasets used for trainingK
dataset_train = Dataset.get_by_name(workspace, name=input_name_train)
dataset_test = Dataset.get_by_name(workspace, name=input_name_test)

# denife output dataset for run metrics JSON file
metrics_output_name = 'metrics_output'
metrics_data = PipelineData(name='metrics_data',
                            datastore=datastore,
                            pipeline_output_name=metrics_output_name)

script_params_1 = [
    '--models', 'deeplearning',
    '--data_folder_train',
    dataset_train_sub.as_named_input('train').as_mount(),
    '--data_folder_test',
    dataset_test_sub.as_named_input('test').as_mount()
]

script_params_2 = [
    '--models', 'deeplearning',
    '--fullmodel', "yes",
    '--data_folder_train',
    dataset_train.as_named_input('train').as_mount(),
    '--data_folder_test',
    dataset_test.as_named_input('test').as_mount()
]

# Define Run Configuration
estimator = PyTorch(
    entry_script='train.py',
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../',
        'modeling'
    ),
    compute_target="alwaysoncluster",
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
        'onnxruntime==1.2.0',
        'onnx==1.6.0'
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
                max_total_runs=2,
                max_concurrent_runs=None
            ),
            estimator_entry_script_arguments=script_params_1,
            outputs=[],
            metrics_output=metrics_data,
            allow_reuse=True,
            version=None
)

# Load run Config
run_config = RunConfiguration.load(
    path=os.path.join(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        "environments/fullmodel_step/RunConfig/runconfig_fullmodel.yml",
        )),
    name="fullmodel"
)


fullmodel = PythonScriptStep(
    name="fullmodel",
    script_name="train.py",
    arguments=script_params_2,
    inputs=[
            metrics_data
    ],
    outputs=[],
    runconfig=run_config_full,
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
experiment = Experiment(workspace, 'pipelinere-train')

# Run the experiment
pipeline_run = experiment.submit(pipeline)
