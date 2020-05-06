"""
Training submitter

Facilitates (remote) training execution through the Azure ML service.
"""
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.runconfig import RunConfiguration
from azureml.train.estimator import Estimator

# Define comfigs
# if data_local = 'yes', then subset should always be 'yes
data_local = "no"  # allowed options: "yes", "no"
subset = "yes"  # allowed options: "yes", "no"

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

if data_local == "yes":
    script_params = {
        '--data_folder_train':
        os.path.join(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../..",
                "outputs/raw_data/", "raw_subset_train.csv",
            )
        ),
        '--data_folder_test':
        os.path.join(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../..",
                "outputs/raw_data/", "raw_subset_test.csv",
            )
        ),
        '--outputfolder':
        os.path.join(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../..",
                "outputs/prepared_data/"
            )
        ),
        '--subset': 'yes',
        '--local': 'yes'
    }
    est = Estimator(
            entry_script='data_engineering.py',
            script_params=script_params,
            source_directory=os.path.dirname(os.path.realpath(__file__)),
            compute_target='local',
            user_managed=True,
            use_docker=False
        )
else:
    # Define datasets names
    if subset == "no":
        # Get environment from config yml for data engineering for full dataset
        filepath = "environments/data_preperation/RunConfig/runconfig_data.yml"
        input_name_train = 'newsgroups_raw_train'
        input_name_test = 'newsgroups_raw_test'
    else:
        # Get environment from config yml for data engineering for full dataset
        filepath = "environments/data_preperation_subset/RunConfig/runconfig_data.yml"
        input_name_train = 'newsgroups_raw_subset_train'
        input_name_test = 'newsgroups_raw_subset_test'

    # Load run Config file for data prep
    run_config = RunConfiguration.load(
        path=os.path.join(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../..",
            filepath,
            )),
        name="dataprep"
    )

    est = ScriptRunConfig(
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        run_config=run_config,
        arguments=[
            '--data_folder_train',
            'DatasetConsumptionConfig:{}'.format(input_name_train),
            '--data_folder_test',
            'DatasetConsumptionConfig:{}'.format(input_name_test),
            '--subset', subset,
            '--local', 'no'
        ],
    )

# Define the ML experiment
experiment = Experiment(workspace, "data-engineering")
# Submit experiment run, if compute is idle, this may take some time')
run = experiment.submit(est)
