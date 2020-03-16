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
    entry_script='train15.py',
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    compute_target='local',
    conda_packages=[
        'pip==20.0.2'
    ],
    pip_packages=[
        'numpy==1.15.4',
        'pandas==0.23.4',
        'scikit-learn==0.20.1',
        'scipy==1.0.0',
        'matplotlib==3.0.2',
        'utils==0.9.0'
    ],
    use_docker=False
)

# Define the ML experiment
experiment = Experiment(workspace, "newsgroups_train_randomforest15")

# Submit experiment run, if compute is idle, this may take some time')
run = experiment.submit(est)

# wait for run completion of the run, while showing the logs
run.wait_for_completion(show_output=True)