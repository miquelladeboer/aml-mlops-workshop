from azureml.core import Workspace, Experiment
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.dataset import Dataset
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import RunConfiguration
import os

# get workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# choose compute target
compute_target = workspace.compute_targets["daskcluster"]

# retrieve subset datasets used for training
subset_dataset_train = Dataset.get_by_name(workspace,
                                           name='newsgroups_subset_train')
subset_dataset_test = Dataset.get_by_name(workspace,
                                          name='newsgroups_subset_test')

# retrieve full datasets used for training full model
# retrieve subset datasets used for trainingK
dataset_train = Dataset.get_by_name(workspace,
                                    name='newsgroups_train')
dataset_test = Dataset.get_by_name(workspace,
                                   name='newsgroups_test')

# Load run Config
run_config = RunConfiguration.load(
    path=os.path.join(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        "environments/data_preperation/RunConfig/runconfig_data_preperation.yml",
        )),
    name="dataprep"
)

dataprep_subset = PythonScriptStep(
    name="subset",
    script_name="data_engineering.py",
    arguments=['--dataset', 'subset_',
                '--data_local', False],
    inputs=[
            dataset_train.as_named_input('raw_subset_train'),
            dataset_test.as_named_input('raw_subset_test'),
    ],
    outputs=[],
    compute_target=compute_target,
    runconfig=run_config,
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'data'
    )
)

dataprep_fulldata = PythonScriptStep(
    name="subset",
    script_name="data_engineering.py",
    arguments=['--dataset', '',
                '--data_local', False],
    inputs=[
            dataset_train.as_named_input('raw_train'),
            dataset_test.as_named_input('raw_test'),
    ],
    outputs=[],
    compute_target=compute_target,
    runconfig=run_config,
    source_directory=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'data'
    )
)

# Attach step to the pipelines
pipeline = Pipeline(workspace=workspace, steps=[dataprep_subset,
                                                dataprep_fulldata])

# Submit the pipeline
# Define the experiment
experiment = Experiment(workspace, 'pipeline data prep')

# Run the experiment
pipeline_run = experiment.submit(pipeline)
