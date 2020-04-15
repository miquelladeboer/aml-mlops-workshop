import os
from azureml.train.estimator import Estimator
from azureml.core.runconfig import (Data,
                                    DataLocation,
                                    Dataset as RunDataset)


class LocalEstimator(Estimator):
    """
    Initiates AML Estimator with local config
    """
    def __init__(self, entry_script, source_directory, script_params):
        super().__init__(
            entry_script='train.py',
            script_params=script_params,
            source_directory=os.path.dirname(os.path.realpath(__file__)),
            compute_target='local',
            user_managed=True,
            use_docker=False
        )


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
