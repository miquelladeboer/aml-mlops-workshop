# Defines a tabular dataset on top of an Azure ML datastore
from azureml.core import Workspace, Datastore, Dataset
from azureml.data import DataType
from azureml.core.authentication import AzureCliAuthentication

# Retrieve a datastore from a ML workspace
ws = Workspace.from_config(auth=AzureCliAuthentication())
datastore_name = 'workspaceblobstore'
datastore = Datastore.get(ws, datastore_name)

# Register dataset and sebset version for each data split
for data_split in ['train', 'test']:
    # Create a TabularDataset from paths in datastore in split folder
    # Note that wildcards can be used
    datastore_paths = [
        (datastore, '{}/*.csv'.format(data_split))
    ]

    # Create a TabularDataset from paths in datastore
    dataset = Dataset.Tabular.from_delimited_files(
        path=datastore_paths,
        set_column_types={
            'text': DataType.to_string(),
            'target': DataType.to_string()
        },
        header=True
    )

    # Register the defined dataset for later use
    dataset.register(
        workspace=ws,
        name='newsgroups_{}'.format(data_split),
        description='newsgroups data',
        create_new_version=True
    )

    # Create a TabularDataset subset from paths in datastore in split folder
    # Note that wildcards can be used
    datastore_paths_subset = [
        (datastore, '{}/*.csv'.format('subset_' + data_split))
    ]

    # Create a TabularDataset from paths in datastore
    dataset_subset = Dataset.Tabular.from_delimited_files(
        path=datastore_paths_subset,
        set_column_types={
            'text': DataType.to_string(),
            'target': DataType.to_string()
        },
        header=True
    )

    # Register the defined dataset for later use
    dataset_subset.register(
        workspace=ws,
        name='newsgroups_{}'.format('subset_' + data_split),
        description='newsgroups data',
        create_new_version=True
    )
