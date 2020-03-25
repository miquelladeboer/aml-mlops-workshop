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
    entry_script='train_15_models.py',
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
experiment = Experiment(workspace, "newsgroups_train_15models")

# Submit experiment run, if compute is idle, this may take some time')
run = experiment.submit(est)

# wait for run completion of the run, while showing the logs
run.wait_for_completion(show_output=True)

# Get the best results
max_run_id = None
max_accuracy = None

for child in run.get_children():
    run_metrics = child.get_metrics()
    run_details = child.get_details()
    # each logged metric becomes a key in this returned dict
    accuracy = run_metrics["accuracy"]
    run_id = run_details["runId"]

    if max_accuracy is None:
        max_accuracy = accuracy
        max_run_id = run_id
    else:
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_runid = run_id

print("Best run_id: " + max_run_id)
print("Best run_id acuuracy: " + str(max_accuracy))
