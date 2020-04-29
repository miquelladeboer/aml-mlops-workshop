"""
Training submitter

Facilitates (remote) training execution through the Azure ML service.
"""
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.runconfig import RunConfiguration

# Define comfigs
data_local = "no"

# Define compute target for data engineering from AML
compute_target = 'alwaysoncluster'

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Define datasets names
# Get environment from config yml for data engineering for full dataset
filepath = "environments/data_profiling/RunConfig/runconfig_data_profiling.yml"
input_name_train = 'newsgroups_raw_subset_train'

# Load run Config file for data prep
run_config = RunConfiguration.load(
    path=os.path.join(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        filepath,
        )),
    name="dataprofiling"
)

est = ScriptRunConfig(
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    run_config=run_config,
    arguments=[
        '--data_folder',
        'DatasetConsumptionConfig:{}'.format(input_name_train),
        '--local', 'no'
    ],
)

# Define the ML experiment
experiment = Experiment(workspace, "historic-profile")
# Submit experiment run, if compute is idle, this may take some time')
run = experiment.submit(est)
