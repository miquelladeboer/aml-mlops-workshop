"""
Training submitter

Facilitates (remote) training execution through the Azure ML service.
"""
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.train.estimator import Estimator
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.runconfig import RunConfiguration

# Define comfigs
data_local = True

# define compute
compute_target = 'alwaysoncluster'

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

if data_local is True:
    # define script parameters
    script_params = {
        '--data_folder_train':
        os.path.join(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "../..",
                    "outputs/raw_data/", "raw_subset_train.csv",
                    )),
        '--data_folder_test':
        os.path.join(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "../..",
                    "outputs/raw_data/", "raw_subset_test.csv",
                    )),
        }
    # Define Run Configuration
    est = Estimator(
        entry_script='train.py',
        script_params=script_params,
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        compute_target='local',
        user_managed=True,
        use_docker=False
    )


if data_local is False:
    # define data set names
    input_name_train = 'newsgroups_raw_subset_train'
    input_name_test = 'newsgroups_raw_subset_test'
    dataset = "subset_"
    filepath = "environments/data_preperation/RunConfig/runconfig_data.yml"

    # Load run Config
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
        arguments=[
            '--data_folder_train',
            'DatasetConsumptionConfig:{}'.format(input_name_train),
            '--data_folder_test',
            'DatasetConsumptionConfig:{}'.format(input_name_test)
        ],
        run_config=run_config
    )

    # Define the ML experiment
    experiment = Experiment(workspace, "explore_sklearn_models")
    # Submit experiment run, if compute is idle, this may take some time')
    run = experiment.submit(est)