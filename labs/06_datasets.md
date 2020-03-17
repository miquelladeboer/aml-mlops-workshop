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

![Getting Started](data-concept-diagram (1).svg)