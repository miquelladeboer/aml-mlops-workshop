from azureml.core import Workspace, Experiment, Datastore
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.dataset import Dataset
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import RunConfiguration
import os
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig, PrimaryMetricGoal)
from azureml.train.dnn import PyTorch
from azureml.core.runconfig import MpiConfiguration
from azureml.pipeline.steps import HyperDriveStep
from azureml.pipeline.core import PipelineData
from azureml.data.data_reference import DataReference

workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Retrieve datastore/datasets
# retrieve datastore
datastore_name = 'workspaceblobstore'
datastore = Datastore.get(workspace, datastore_name)

# data reference
baseline_profile = DataReference(
    datastore,
    data_reference_name='baselineProfile',
    path_on_datastore='baseline_profile',
    mode='download',
    path_on_compute=None,
    overwrite=False
)

# data reference
historic_profile = DataReference(
    datastore,
    data_reference_name='historicProfile',
    path_on_datastore='historic_profile',
    mode='download',
    path_on_compute=None,
    overwrite=False
)

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

train_validated = PipelineData(
    name='newsgroups_raw_train',
    datastore=datastore,
    pipeline_output_name='newsgroups_validated_train'
)

test_validated = PipelineData(
    name='newsgroups_raw_test',
    datastore=datastore,
    pipeline_output_name='newsgroups_validated_test'
)

subset_train_validated = PipelineData(
    name='newsgroups_raw_subset_train',
    datastore=datastore,
    pipeline_output_name='newsgroups_validated__subsettrain'
)

subset_test_validated = PipelineData(
    name='newsgroups_raw_subset_test',
    datastore=datastore,
    pipeline_output_name='newsgroups_validated_subset_test'
)

subset_train_prepared = PipelineData(
    name='newsgroups_subset_train',
    datastore=datastore,
    pipeline_output_name='newsgroups_subset_train'
)

subset_test_prepared = PipelineData(
    name='newsgroups_subset_test',
    datastore=datastore,
    pipeline_output_name='newsgroups_subset_test'
)

train_prepared = PipelineData(
    name='newsgroups_train',
    datastore=datastore,
    pipeline_output_name='newsgroups_train'
)

test_prepared = PipelineData(
    name='newsgroups_test',
    datastore=datastore,
    pipeline_output_name='newsgroups_test'
)

# denife output dataset for run metrics JSON file
metrics_output_name = 'metrics_output'
metrics_data = PipelineData(
    name='metrics_data',
    datastore=datastore,
    pipeline_output_name=metrics_output_name
)

modelpath_name = 'modelpath'
modelpath = PipelineData(
    name='modelpath',
    datastore=datastore,
    pipeline_output_name=modelpath_name
)

sklearnmodelpath_name = 'sklearnmodel'
sklearnmodelpath = PipelineData(
    name='sklearnmodelpath',
    datastore=datastore,
    pipeline_output_name=sklearnmodelpath_name
)

historicprofile_name = 'historicprofile'
historicprofile = PipelineData(
    name='historicprofile',
    datastore=datastore,
    pipeline_output_name=historicprofile_name
)

datadrift_name = 'data_drift_report'
datadriftreport = PipelineData(
    name='data_drift_report',
    datastore=datastore,
    pipeline_output_name=datadrift_name
)

datadrift_subset_name = 'data_drift_report_subset'
datadriftreportsubset = PipelineData(
    name='data_drift_report_subset',
    datastore=datastore,
    pipeline_output_name=datadrift_subset_name
)

# Set parameters for search
param_sampling = BayesianParameterSampling({
    "learning_rate": uniform(10e-6, 1e0),
    "num_epochs": choice(10, 20),
    "batch_size": choice(10, 20, 50, 100, 200, 300, 500, 1000),
    "hidden_size": choice(300, 400)
})

# LOAD ALL SCRIPT PARAMETERS FOR EVERY STEP IN PIPELINE
script_params_data_validation = [
    '--data_folder_train',
    dataset_train.as_named_input('train').as_mount(),
    '--data_folder_test',
    dataset_test.as_named_input('test').as_mount(),
    '--local', 'no',
    '--output_train', train_validated,
    '--output_test', test_validated,
    '--data_drift_report', datadriftreport
]

script_params_data_validation_subset = [
    '--data_folder_train',
    dataset_train_sub.as_named_input('train').as_mount(),
    '--data_folder_test',
    dataset_test_sub.as_named_input('test').as_mount(),
    '--local', 'no',
    '--output_train', subset_train_validated,
    '--output_test', subset_test_validated,
    '--data_drift_report', datadriftreportsubset
]

script_params_historic_profile = [
    '--local', 'no',
    '--new_profile', 'no',
    '--input_train', train_validated,
    '--historicprofile', historicprofile
]

script_params_data_engineering_subset = [
    '--subset', 'yes',
    '--local', 'no',
    "--input_train", subset_train_validated,
    "--input_test", subset_test_validated,
    "--output_train", subset_train_prepared,
    "--output_test", subset_test_prepared
    ]

script_params_data_engineering = [
    '--subset', 'no',
    '--local', 'no',
    "--input_train", train_validated,
    "--input_test", test_validated,
    "--output_train", train_prepared,
    "--output_test", test_prepared
    ]

script_params_hyperdrive = [
    '--models', 'deeplearning',
    '--input_train', subset_train_prepared,
    '--input_test', subset_test_prepared
]

script_params_sklearn = [
    '--models', 'sklearnmodels',
    '--input_train', subset_train_prepared,
    '--input_test', subset_test_prepared,
    '--sklearnmodel', sklearnmodelpath
]

script_params_model_training = [
    '--models', 'deeplearning',
    '--fullmodel', "yes",
    '--input_train', train_prepared,
    '--input_test', test_prepared,
    '--savemodel',  modelpath
]

# LOAD RUN CONFIGURATIONS FOR EVERY STEP IN THE PIPELINE
# Load run configuration for data validation on full dataset
filepath = "environments/data_validation/RunConfig/runconfig_data_validation.yml"
run_config_data_validation = RunConfiguration.load(
    path=os.path.join(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        filepath,
        )),
    name="datavalidation"
)

# Load run configuration for data validation on subset
filepath = "environments/data_validation_subset/RunConfig/runconfig_data_validation.yml"
run_config_data_validation_subset = RunConfiguration.load(
    path=os.path.join(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        filepath,
        )),
    name="datavalidation"
)

# load run configuration for historic profiling
filepath = "environments/data_profiling/RunConfig/runconfig_data_profiling.yml"
# Load run Config file for data prep
run_config_historic_profile = RunConfiguration.load(
    path=os.path.join(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        filepath,
        )),
    name="dataprofiling"
)

# Load run Config data enhineering subset
run_config_data_engineering_subset = RunConfiguration.load(
 path=os.path.join(os.path.join(
  os.path.dirname(os.path.realpath(__file__)),
  "../..",
  "environments/data_preperation_subset/RunConfig/runconfig_data.yml",
  )),
 name="dataprep_subset"
)

# Load run Config data engineering entire dataset
run_config_data_engineering = RunConfiguration.load(
 path=os.path.join(os.path.join(
  os.path.dirname(os.path.realpath(__file__)),
  "../..",
  "environments/data_preperation/RunConfig/runconfig_data.yml",
  )),
 name="dataprep_full"
)

# Load run Config
run_config_sklearn = RunConfiguration.load(
    path=os.path.join(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        "environments/sklearn_subset/RunConfig/runconfig_subset.yml",
        )),
    name="sklearn"
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

# DEFINE ALL PIPELINE STEPS
# data validation for enitre dataset
data_validation = PythonScriptStep(
    name="data_validation",
    script_name="data_validation.py",
    arguments=script_params_data_validation,
    runconfig=run_config_data_validation,
    inputs=[
        baseline_profile
    ],
    outputs=[
        train_validated,
        test_validated,
        datadriftreport
    ],
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'data validation and preparation'
    )
)
# data validation for subset dataset
data_validation_subset = PythonScriptStep(
    name="data_validation_subset",
    script_name="data_validation.py",
    arguments=script_params_data_validation_subset,
    runconfig=run_config_data_validation_subset,
    inputs=[
        baseline_profile
    ],
    outputs=[
        subset_train_validated,
        subset_test_validated,
        datadriftreportsubset
    ],
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'data validation and preparation'
    )
)

# create hsitoric profile
historic_profile = PythonScriptStep(
    name="historic_profile",
    script_name="create_historic_profile.py",
    arguments=script_params_historic_profile,
    inputs=[
            train_validated,
            historic_profile
    ],
    outputs=[
             historicprofile
    ],
    runconfig=run_config_historic_profile,
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'data validation and preparation'
    )
)

# data engineering on subset data
data_engineering_subset = PythonScriptStep(
    name="data_engineering_subset",
    script_name="data_engineering.py",
    arguments=script_params_data_engineering_subset,
    runconfig=run_config_data_engineering_subset,
    inputs=[
        subset_train_validated,
        subset_test_validated
    ],
    outputs=[
        subset_train_prepared,
        subset_test_prepared
    ],
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'data validation and preparation'
    )
)

# data engineering all data
data_engineering = PythonScriptStep(
    name="data_engineering",
    script_name="data_engineering.py",
    arguments=script_params_data_engineering,
    runconfig=run_config_data_engineering,
    inputs=[
        train_validated,
        test_validated
    ],
    outputs=[
        train_prepared,
        test_prepared
    ],
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'data validation and preparation'
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
            estimator_entry_script_arguments=script_params_hyperdrive,
            inputs=[
                subset_train_prepared,
                subset_test_prepared
            ],
            outputs=[],
            metrics_output=metrics_data,
            allow_reuse=True,
            version=None
)

sklearn_models = PythonScriptStep(
    name="sklearn",
    script_name="train.py",
    arguments=script_params_sklearn,
    inputs=[
            subset_train_prepared,
            subset_test_prepared
    ],
    outputs=[
        sklearnmodelpath
    ],
    runconfig=run_config_sklearn,
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'modeling'
    )
)

model_training = PythonScriptStep(
    name="fullmodel",
    script_name="train.py",
    arguments=script_params_model_training,
    inputs=[
            metrics_data,
            train_prepared,
            test_prepared
    ],
    outputs=[
             modelpath
    ],
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
        data_validation,
        data_validation_subset,
        historic_profile,
        data_engineering_subset,
        data_engineering,
        hypertuning,
        model_training,
        sklearn_models
    ]
)

# Submit the pipeline
# Define the experiment
experiment = Experiment(workspace, 'pipeline-retrain-model')

# Run the experiment
pipeline_run = experiment.submit(pipeline)
