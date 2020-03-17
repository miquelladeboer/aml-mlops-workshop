"""
Training submitter

Facilitates (remote) training execution through the Azure ML service.
"""
import os
from azureml.core import Workspace, Experiment
from azureml.train.estimator import Estimator
from azureml.core.authentication import AzureCliAuthentication

# load Azure ML workspace


# Define Run Configuration


# Define the ML experiment


# Submit experiment run, if compute is idle, this may take some time')

# wait for run completion of the run, while showing the logs
