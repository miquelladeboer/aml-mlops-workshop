from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig, PrimaryMetricGoal)
from azureml.core import Workspace, Experiment
from azureml.train.estimator import Estimator
import os
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
from azureml.core.authentication import AzureCliAuthentication

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Define Run Configuration

# Set parameters for search

# Define multi-run configuration

# Define the ML experiment

# Submit the experiment

# Select the best run from all submitted

# Log the best run's performance to the parent run

# Best set of parameters found
