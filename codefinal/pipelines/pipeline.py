from azureml.core import Workspace, Experiment
from azureml.train.estimator import Estimator
import os
from azureml.train.hyperdrive.parameter_expressions import uniform, choice

from azureml.pipeline.core import PipelineData

from azureml.core.authentication import AzureCliAuthentication
from azureml.core.dataset import Dataset

from azureml.data.data_reference import DataReference

from azureml.pipeline.steps import PythonScriptStep, HyperDriveStep
from azureml.pipeline.core import Pipeline

from azureml.train.dnn import PyTorch

from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig, PrimaryMetricGoal)
from azureml.core import Workspace, Experiment
from azureml.core.runconfig import MpiConfiguration
from azureml.train.estimator import Estimator
import os
from azureml.train.hyperdrive.parameter_expressions import uniform, choice

from  azureml.data.dataset_consumption_config import DatasetConsumptionConfig

from azureml.core import Workspace, Datastore, Dataset
from azureml.data import DataType

workspace = Workspace.from_config(auth=AzureCliAuthentication())

# retrieve datastore
datastore_name = 'workspaceblobstore'
datastore = Datastore.get(workspace, datastore_name)

# retrieve datasets used for training
subset_dataset_train = Dataset.get_by_name(workspace,
                                           name='newsgroups_subset_train')
subset_dataset_test = Dataset.get_by_name(workspace,
                                          name='newsgroups_subset_test')

# retrieve datasets used for training
dataset_train = Dataset.get_by_name(workspace, name='newsgroups_train')
dataset_test = Dataset.get_by_name(workspace, name='newsgroups_test')

compute_target_hyper = workspace.compute_targets["hypercomputegpu"]
compute_target_fullmodel = workspace.compute_targets["fullcomputegpu"]

# Define Run Configuration
estimator = PyTorch(
    entry_script='train.py',
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'modeling'
    ),
    compute_target=compute_target_hyper,
    distributed_training=MpiConfiguration(),
    framework_version='1.4',
    use_gpu=True,
    pip_packages=[
        'numpy==1.15.4',
        'pandas==0.23.4',
        'scikit-learn==0.20.1',
        'scipy==1.0.0',
        'matplotlib==3.0.2',
        'utils==0.9.0',
    ],
    inputs=[
        dataset_train.as_named_input('subset_train'),
        dataset_train.as_named_input('subset_test')
    ]
)

# Set parameters for search
param_sampling = BayesianParameterSampling({
    "learning_rate": uniform(0.05, 0.1),
    "num_epochs": choice(5, 10, 15),
    "batch_size": choice(150, 200),
    "hidden_size": choice(50, 100)
})

experiment = Experiment(workspace, 're-train')


metrics_output_name = 'metrics_output'
metrics_data = PipelineData(name='metrics_data',
                            datastore=datastore,
                            pipeline_output_name=metrics_output_name)

hypertrain = HyperDriveStep(
            name='hypertrain',
            hyperdrive_config=HyperDriveConfig(
                estimator=estimator,
                hyperparameter_sampling=param_sampling,
                policy=None,
                primary_metric_name="accuracy",
                primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                max_total_runs=80,
                max_concurrent_runs=None
            ),
            estimator_entry_script_arguments=[],
            inputs=[
                    dataset_train.as_named_input('subset_train'),
                    dataset_test.as_named_input('subset_test')
                    ],
            outputs=[],
            metrics_output=metrics_data,
            allow_reuse=True,
            version=None
)

pipeline = Pipeline(workspace=workspace, steps=hypertrain)

pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion()
