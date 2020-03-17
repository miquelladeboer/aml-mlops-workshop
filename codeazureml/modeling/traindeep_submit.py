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
estimator = Estimator(
    entry_script='traindeep.py',
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    compute_target='local',
    pip_packages=[
        'numpy==1.15.4',
        'pandas==0.23.4',
        'scikit-learn==0.20.1',
        'scipy==1.0.0',
        'matplotlib==3.0.2',
        'utils==0.9.0',
        'torch==1.4.0'
    ]
)

# Set parameters for search
param_sampling = BayesianParameterSampling({
    "learning_rate": uniform(0.05, 0.1),
    "num_epochs": choice(5, 10, 15),
    "batch_size": choice(150, 200),
    "hidden_size": choice(50, 100)
})

# Define multi-run configuration
hyperdrive_run_config = HyperDriveConfig(
    estimator=estimator,
    hyperparameter_sampling=param_sampling,
    policy=None,
    primary_metric_name="accuracy",
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=80,
    max_concurrent_runs=None
)

# Define the ML experiment
experiment = Experiment(workspace, "newsgroups_train_hypertune")

# Submit the experiment
hyperdrive_run = experiment.submit(hyperdrive_run_config)
hyperdrive_run.wait_for_completion()

# Select the best run from all submitted
best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()

# Log the best run's performance to the parent run
hyperdrive_run.log("Accuracy", best_run_metrics['accuracy'])
parameter_values = best_run.get_details()['runDefinition']['arguments']

# Best set of parameters found
best_parameters = dict(zip(parameter_values[::2], parameter_values[1::2]))
best_model_parameters = best_parameters.copy()


## Define a final training run with model's best parameters
#model_est = Estimator(
#    entry_script='hypertrain.py',
#    source_directory=os.path.dirname(os.path.realpath(__file__)),
#    script_params=best_model_parameters,
#    compute_target='local',
#    pip_packages=[
#        'numpy==1.15.4',
#        'pandas==0.23.4',
#        'scikit-learn==0.20.1',
#        'scipy==1.0.0',
#        'matplotlib==3.0.2',
#        'utils==0.9.0'
#    ]
#)

# Submit the experiment
#model_run = experiment.submit(model_est)

#model_run_status = model_run.wait_for_completion(wait_post_processing=True)

#model = model_run.register_model(model_name='model',
#                                 model_path=os.path.join('outputs', 'model.pkl'))
