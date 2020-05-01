"""
Training submitter

Facilitates (remote) training execution through the Azure ML service.
"""
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.runconfig import RunConfiguration
from azureml.core.dataset import Dataset

# parameter switch
subset = "yes"

# Define compute target for data engineering from AML
compute_target = 'alwaysoncluster'

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Define datasets names
if subset == "no":
    # Get environment from config yml for data engineering for full dataset
    filepath = "environments/data_validation/RunConfig/runconfig_data_validation.yml"
    input_name_train = 'newsgroups_raw_train'
    input_name_test = 'newsgroups_raw_test'
else:
    # Get environment from config yml for data engineering for full dataset
    filepath = "environments/data_validation_subset/RunConfig/runconfig_data_validation.yml"
    input_name_train = 'newsgroups_raw_subset_train'
    input_name_test = 'newsgroups_raw_subset_test'

dataset_train = Dataset.get_by_name(workspace, name=input_name_train)
dataset_test = Dataset.get_by_name(workspace, name=input_name_test)


# Load run Config file for data prep
run_config = RunConfiguration.load(
    path=os.path.join(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        filepath,
        )),
    name="datavalidation"
)

est = ScriptRunConfig(
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    run_config=run_config,
    arguments=[
        '--data_folder_train',
        dataset_train.as_named_input('train').as_mount(),
        '--data_folder_test',
        dataset_test.as_named_input('test').as_mount(),
        '--local', 'no'
    ],
)

# Define the ML experiment
experiment = Experiment(workspace, "data-validation")
# Submit experiment run, if compute is idle, this may take some time')
run = experiment.submit(est)
