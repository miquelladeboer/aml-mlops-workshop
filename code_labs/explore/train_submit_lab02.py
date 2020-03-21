"""
Training submitter

Facilitates (remote) training execution through the Azure ML service.
"""
import os
from azureml.core import Workspace, Experiment
from azureml.train.estimator import Estimator
from azureml.core.authentication import AzureCliAuthentication

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Define Run Configuration
est = Estimator(
    entry_script='train.py',
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    compute_target='local',
    conda_dependencies_file=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../',
        'conda_dependencies.yml'
    ),
    use_docker=False
)

# Define the ML experiment
experiment = Experiment(workspace, "newsgroups_train_randomforest")

# Submit experiment run, if compute is idle, this may take some time')
run = experiment.submit(est)

# wait for run completion of the run, while showing the logs
run.wait_for_completion(show_output=True)
