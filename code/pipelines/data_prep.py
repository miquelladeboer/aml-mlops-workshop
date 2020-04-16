from azureml.core import Workspace, Experiment, Datastore
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.dataset import Dataset
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import RunConfiguration
import os

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

# define script parameters
script_params_sub = [
    '--data_folder_train',
    dataset_train_sub.as_named_input('train').as_mount(),
    '--data_folder_test',
    dataset_test_sub.as_named_input('test').as_mount(),
    '--subset', True,
    '--local', False
    ]

# define script parameters
script_params = [
    '--data_folder_train',
    dataset_train.as_named_input('train').as_mount(),
    '--data_folder_test',
    dataset_test.as_named_input('test').as_mount(),
    '--subset', True,
    '--local', False
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

dataprep_subset = PythonScriptStep(
    name="subset",
    script_name="data_engineering.py",
    arguments=script_params_sub,
    runconfig=run_config_sub,
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
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'data'
    )
)

# Attach step to the pipelines
pipeline = Pipeline(workspace=workspace, steps=[dataprep_subset,
                                                dataprep_fulldata])

# Submit the pipeline
# Define the experiment
experiment = Experiment(workspace, 'pipeline-dataprep')

# Run the experiment
pipeline_run = experiment.submit(pipeline)
