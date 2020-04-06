from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import MpiConfiguration, RunConfiguration
import os

# Define the conda dependencies
cd = CondaDependencies(
    conda_dependencies_file_path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'conda_dependencies_pytorch.yml'
    )
)
# Runconfig
amlcompute_run_config = RunConfiguration(conda_dependencies=cd,
                                         communicator=MpiConfiguration(),
                                         framework='PyTorch')

amlcompute_run_config.environment.docker.enabled = True
amlcompute_run_config.environment.spark.precache_packages = False


amlcompute_run_config.save(path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "RunConfig",
        "runconfig_pytorch.yml",
    ), name='pytorch', separate_environment_yaml=True)
