from azureml.core import Workspace, Experiment, Datastore
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.dataset import Dataset
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import RunConfiguration
import os
import argparse
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig, PrimaryMetricGoal)
from azureml.train.dnn import PyTorch
from azureml.core.runconfig import MpiConfiguration
from azureml.pipeline.steps import HyperDriveStep
from azureml.pipeline.core import PipelineData

parser = argparse.ArgumentParser()
parser.add_argument("--await_completion",
                    type=bool,
                    default=False)
parser.add_argument("--download_outputs",
                    type=bool,
                    default=False)
args = parser.parse_args()

workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Set parameters for search
param_sampling = BayesianParameterSampling({
    "learning_rate": uniform(10e-6, 1e0),
    "num_epochs": choice(10, 20),
    "batch_size": choice(10, 20, 50, 100, 200, 300, 500, 1000),
    "hidden_size": choice(300, 400)
})

# Retrieve datastore/datasets
# retrieve datastore
datastore_name = 'workspaceblobstore'
datastore = Datastore.get(workspace, datastore_name)

# define data set names
input_name_train_sub = 'newsgroups_raw_subset_train'
input_name_test_sub = 'newsgroups_raw_subset_test'

input_name_train = 'newsgroups_raw_train'
input_name_test = 'newsgroups_raw_test'

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

datastore = workspace.get_default_datastore()
subset_train = PipelineData(name='newsgroups_subset_train',
                            datastore=datastore,
                            pipeline_output_name='newsgroups_subset_train')
subset_test = PipelineData(name='newsgroups_subset_test',
                           datastore=datastore,
                           pipeline_output_name='newsgroups_subset_test')
train = PipelineData(name='newsgroups_train',
                     datastore=datastore,
                     pipeline_output_name='newsgroups_train')
test = PipelineData(name='newsgroups_test',
                    datastore=datastore,
                    pipeline_output_name='newsgroups_test')

# define script parameters
script_params_sub = [
    '--data_folder_train',
    dataset_train_sub.as_named_input('train').as_mount(),
    '--data_folder_test',
    dataset_test_sub.as_named_input('test').as_mount(),
    '--subset', 'yes',
    '--local', 'no',
    "--output_train", subset_train,
    "--output_test", subset_test
    ]

# define script parameters
script_params = [
    '--data_folder_train',
    dataset_train.as_named_input('train').as_mount(),
    '--data_folder_test',
    dataset_test.as_named_input('test').as_mount(),
    '--subset', 'no',
    '--local', 'no',
    "--output_train", train,
    "--output_test", test
    ]

script_params_1 = [
    '--models', 'deeplearning',
    '--pipeline', 'yes',
    '--output_train', subset_train,
    '--output_test', subset_test
]

script_params_2 = [
    '--models', 'deeplearning',
    '--fullmodel', "yes",
    '--pipeline', 'yes',
    '--output_train', train,
    '--output_test', test
]

# Load run Config
run_config_sub = RunConfiguration.load(
 path=os.path.join(os.path.join(
  os.path.dirname(os.path.realpath(__file__)),
  "../..",
  "environments/data_preperation_subset/RunConfig/runconfig_data.yml",
  )),
 name="dataprep_subset"
)

# Load run Config
run_config = RunConfiguration.load(
 path=os.path.join(os.path.join(
  os.path.dirname(os.path.realpath(__file__)),
  "../..",
  "environments/data_preperation/RunConfig/runconfig_data.yml",
  )),
 name="dataprep_full"
)

# Load run Config
run_config_full = RunConfiguration.load(
    path=os.path.join(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        "environments/fullmodel_step/RunConfig/runconfig_fullmodel.yml",
        )),
    name="fullmodel"
)

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

dataprep_subset = PythonScriptStep(
    name="subset",
    script_name="data_engineering.py",
    arguments=script_params_sub,
    runconfig=run_config_sub,
    outputs=[subset_train, subset_test],
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'data'
    )
)

dataprep_fulldata = PythonScriptStep(
    name="full",
    script_name="data_engineering.py",
    arguments=script_params,
    runconfig=run_config,
    outputs=[train, test],
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'data'
    )
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
            inputs=[subset_train, subset_test],
            outputs=[],
            metrics_output=metrics_data,
            allow_reuse=True,
            version=None
)

fullmodel = PythonScriptStep(
    name="fullmodel",
    script_name="train.py",
    arguments=script_params_2,
    inputs=[
            metrics_data,
            train,
            test
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
pipeline = Pipeline(
    workspace=workspace,
    steps=[
        dataprep_subset,
        dataprep_fulldata,
        hypertuning,
        fullmodel
    ]
)

# Submit the pipeline
# Define the experiment
experiment = Experiment(workspace, 'pipeline-retrain-model')

# Run the experiment
pipeline_run = experiment.submit(pipeline)

# Wait for completion if arg provided e.g. for CI scenarios
if args.await_completion is True:
    pipeline_run.wait_for_completion()

    if args.download_outputs is True:
        step_run = pipeline_run.find_step_run("fullmodel")[0]
        print("outputs: {}".format(step_run.get_outputs()))

    port_data_reference = step_run.get_output_data("models")
    port_data_reference.download(local_path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../',
        'outputs/models/'
    ))
