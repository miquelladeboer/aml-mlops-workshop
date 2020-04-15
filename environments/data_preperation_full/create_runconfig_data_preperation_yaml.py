from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.dataset import Dataset
import os

from azureml.core.runconfig import (Data,
                                    DataLocation,
                                    Dataset as RunDataset)


def load_data(dataset, input_name):
    data = Data(
        data_location=DataLocation(
            dataset=RunDataset(dataset_id=dataset.id)),
        create_output_directories=False,
        mechanism='mount',
        environment_variable_name=input_name,
        overwrite=True
        )
    return data


workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Define the conda dependencies
cd = CondaDependencies(
    conda_dependencies_file_path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'conda_dependencies_data_preperation.yml'
    )
)

# define compute
compute_target = 'alwaysoncluster'

# define data set names
input_name_train = 'newsgroups_train'
input_name_test = 'newsgroups_test'

# Retrieve datsets
dataset_train = Dataset.get_by_name(workspace, name=input_name_train)
dataset_test = Dataset.get_by_name(workspace, name=input_name_test)

# Runconfig
amlcompute_run_config = RunConfiguration( conda_dependencies=cd)
    
amlcompute_run_config.environment.docker.enabled = True
amlcompute_run_config.environment.spark.precache_packages = False
amlcompute_run_config.target = compute_target
amlcompute_run_config.data = {input_name_train: load_data(dataset_train,
                                                          input_name_train),
                              input_name_test: load_data(dataset_test,
                                                         input_name_test)}

amlcompute_run_config.save(path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "RunConfig/",
        "runconfig_data_preperation.yml",
    ), name='dataprep_full', separate_environment_yaml=True)