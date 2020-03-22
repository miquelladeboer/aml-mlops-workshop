## Lab 6: Datsets ##
In the previous steps we have seen how we can leverage HyperDrive to select the best hyperparameters for out Neural Network. As we have seen, hyperparameter tuning is a very intensive job. Therefore we want only to use a subset of our dataset for tuning and train the final model with the best hyperparameters with the entire dataset.

In this lab, we are going to see how we can levarge Azure ML datsets following
- create subsets of the data for hypertraining
- use total dataset for final model

## Understand the concepts 
Access data in storage
To access your data in your storage account, Azure Machine Learning offers datastores and datasets. Datastores answer the question: how do I securely connect to my data that's in my Azure Storage? Datastores save the connection information to your Azure Storage. This aids in security and ease of access to your storage, since connection information is kept in the datastore and not exposed in scripts.

Datasets answer the question: how do I get specific data files in my datastore? Datasets point to the specific file or files in your underlying storage that you want to use for your machine learning experiment. Together, datastores and datasets offer a secure, scalable, and reproducible data delivery workflow for your machine learning tasks.

Datastores
An Azure Machine Learning datastore keeps the connection information to your storage so you don't have to code it in your scripts. Register and create a datastore to easily connect to your Azure storage account, and access the data in your underlying Azure storage services.

Datasets
Create an Azure Machine Learning dataset to interact with data in your datastores and package your data into a consumable object for machine learning tasks. Register the dataset to your workspace to share and reuse it across different experiments without data ingestion complexities.

Datasets can be created from local files, public urls, Azure Open Datasets, or specific file(s) in your datastores. To create a dataset from an in memory pandas dataframe, write the data to a local file, like a csv, and create your dataset from that file. Datasets aren't copies of your data, but are references that point to the data in your storage service, so no extra storage cost is incurred.

![alt text](https://docs.microsoft.com/en-us/azure/machine-learning/media/concept-data/data-concept-diagram.svg)

For more information check: https://docs.microsoft.com/en-us/azure/machine-learning/concept-data?view=azure-ml-py

## Ingest data into Datastore ##
Before we can make use of Datastores and Datasets, we need to have our data into the Datastores. In this tuturial, we are going to make use of the Public URL, as our data is available there through the 20newsgroups from sklearn.

1. Open the script `code\Data_Acquisition_and_Understanding\ingest_data.py`
    In this file we are performing the following steps:
    - Load 20newgroups data from sklearn
    - Construct pandas data frame from loaded sklearn newsgroup data
    - pre-process: remove line breaks and replace target index by newsgroup name
    - write to csv

Now that we have preprocessed our data and have it in readable csv files, we want to upload the files to the Datastore by refactoring the script `ingest_data.py`.

2. Import the requited libraries
    ```
    from azureml.core import Workspace, Datastore
    from azureml.core.authentication import AzureCliAuthentication
    ```
3. Define a Datastore name
    `datastore_name = 'workspaceblobstore'`
workspaceblobstore is a Datastore that is by default already avaiable in the Azure ML workspace.

4. Get existing ML workspace
    `workspace = Workspace.from_config(auth=AzureCliAuthentication())`

5. Retrieve an existing datastore in the workspace by name
    `datastore = Datastore.get(workspace, datastore_name)`

6. Upload the csv files to the data store
    ```
    # upload files
    datastore.upload(
        src_dir=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'tmp'
        ),
        target_path=None,
        overwrite=True,
        show_progress=True
    )
    ```

7. Run the script 

8. Go to the Azure Portal to inspect the Datastore
    - Go to the Azure Portal
    - Navigate to the resource group azuremlworkshoprgp
    - Navigate to your storage account
    - Navigate to containers
    - Navigate to azureml-blobstore- ,<ID>
    - In here we see the two created folders train and test that contain the csv files

Note: the correct code is already available in `codeazureml\Data_Acquisition_and_Understanding\ingest_data.py`. In here, all ready to use code is available for the entire workshop.

## Define the Dataset ##
Now that we have out data in the datastore, we want to create datasets. For each version of the train and test data that we have, we want to register it under a new version. This is something that Azure ML does for us. This way we van easily manage data versions and when new data comes in, we know exactly which data is used for training our model. In this way, we can also create different datasets for different puposes of training. In our case, we are using a different data set (a subset) for hyperparamet tuning and only the entire data set for the final model.

1. Open the script `code\Data_Acquisition_and_Understanding\define_dataset.py`
    In this file, we take 3 steps:
    - Get the Datastore path
    - Create a TabularDataset from paths in datastore
    - Register the defined dataset for later use

Refactor the code in `code\Data_Acquisition_and_Understanding\define_dataset.py` :

2. Defines a tabular dataset on top of an Azure ML datastore
    ```
    from azureml.core import Workspace, Datastore, Dataset
    from azureml.data import DataType
    from azureml.core.authentication import AzureCliAuthentication
    ```
3. Retrieve a datastore from a ML workspace
    ```
    ws = Workspace.from_config(auth=AzureCliAuthentication())
    datastore_name = 'workspaceblobstore'
    datastore = Datastore.get(ws, datastore_name)
    ```

4. Define the datsets
    By adding the following code, we add the steps mentioned in step 1
    ```
    # Register dataset and sebset version for each data split
    for data_split in ['train', 'test']:
        # Create a TabularDataset from paths in datastore in split folder
        # Note that wildcards can be used
        datastore_paths = [
            (datastore, '{}/*.csv'.format(data_split))
        ]

        # Create a TabularDataset subset from paths in datastore in split folder
        # Note that wildcards can be used
        datastore_paths_subset = [
            (datastore, '{}/*.csv'.format('subset_' + data_split))
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
            dataset.register(
            workspace=ws,
            name='newsgroups_{}'.format('subset_' + data_split),
            description='newsgroups data',
            create_new_version=True
        )

        # Register the defined dataset for later use
        dataset_subset.register(
            workspace=ws,
            name='newsgroups_{}'.format('subset_' + data_split),
            description='newsgroups data',
            create_new_version=True
        )
        ```
5. Run the script   

6. Go to the Azure ML Workspace and review youe Datasets

Note: the correct code is already available in `codeazureml\Data_Acquisition_and_Understanding\define_dataset.py`. In here, all ready to use code is available for the entire workshop.

## Refactor train script to work with datasets ##
We need to refactor the `traindeep.py` file to work with the datasets from Azure ML and we need to alter the `traindeep_submit.py` file to include the Datasets as arguments in the estimator.

## ALter the deeptrain.py file

1. Load the input datasets from Azure ML
    Load the Tabular datasets from Azure ML and covert them to a Pandas dataframe:
    ```
    dataset_train = run.input_datasets['subset_train'].to_pandas_dataframe()
    dataset_test = run.input_datasets['subset_test'].to_pandas_dataframe()
    ```

2.  Pre-process
    ```
    class data_train(object):
        def __init__(self, data, target):
            self.data = []
            self.target = []


    class data_test(object):
        def __init__(self, data, target):
            self.data = []
            self.target = []

    # convert to numpy df
    data_train.data = dataset_train.text.values
    data_test.data = dataset_test.text.values

    # convert label to int
    data_train.target = [int(value or 0) for value in dataset_train.target.values]
    data_test.target = [int(value or 0) for value in dataset_test.target.values]

## ALter the deeptrain_hyper_submit.py file
We are not going to refactor the `deeptrain_hyper_submit.py` file to give arguments to out estimator when we submit the run.
1. Retrieve datasets used for training
    ```
    dataset_train = Dataset.get_by_name(workspace, name='newsgroups_subset_train')
    dataset_test = Dataset.get_by_name(workspace, name='newsgroups_subset_test')
    ```
2. Set the Datasets as arguments
    In:
    ```
    # Define Run Configuration
    estimator = PyTorch(
        entry_script='traindeep.py',
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        compute_target=workspace.compute_targets[gpu_cluster_name],
        distributed_training=MpiConfiguration(),
        framework_version='1.4',
        use_gpu=True,
        pip_packages=[
            'numpy==1.15.4',
            'pandas==0.23.4',
            'scikit-learn==0.20.1',
            'scipy==1.0.0',
            'matplotlib==3.0.2',
            'utils==0.9.0'
        ]
    )
    ```
    Add arguments:
    ```
    inputs=[
        dataset_train.as_named_input('subset_train'),
        dataset_train.as_named_input('subset_test')
    ]
    ```





