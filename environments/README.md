# Run Configurations

This folder contains configurations for different steps of datapreperation, hypertuning and modeling. These run configurations consist of:
* Conda dependencies file 
* Compute target
* Data sets
* Framework (e.g. Python, Pytorch, Tesonflow, Keras ect.)

As we can understand, for each part of the progress we might need slighly different run config files. For example, we want to do hypertuning only on a subset of the data to save time/money ect, but want to train the full model on the enitre dataset. We might want to use different compute as well for different parts in the process. For example, we might want to do our heavy data transformations on a sprak cluster for distributed data tranformations. But to train our full model, we might want to use gpu compute to imrove the training performance and fit all the data and the model on one cluster. We might also use different conda dependencies in different situations. For example, if we are preparing our data, we might want to use libraries as Dask or Fastpparquet, whereas in model training we might need a Pytorch framework or onnx for model packaging. In order to make our ML steps as clean as possible, we create seperated config files, that only contain the configuration setting needed for a specific ML step.

## Configure the run configurations

In this example we make use of 5 different run configuration files:
* data_preparation_subset
    * Condadependencies file: conda_dependencies_data_preperation.yml, including dask, pysprark, azureml-sdk[notebooks]
    * Compute Target: Small AML compute
    * Data sets: subset
    * Framework: Python
* data_preparation, including dask, pysprark, azureml-sdk[notebooks]
    * Condadependencies file: conda_dependencies_data_preperation.yml
    * Compute Target: Spark cluster
    * Data sets: entire
    * Framework: Python
* sklearn_subset
    * Condadependencies file: conda_dependencies_sklearn.yml, including sklearn and matplotlib
    * Compute Target: 1 cpu cluster
    * Data sets: subset
    * Framework: Python
* sklearn_full
    * Condadependencies file: conda_dependencies_sklearn.yml, includigng sklearn and matplotlib
    * Compute Target: 1 cpu cluster
    * Data sets: entire
    * Framework: Python
* fullmodel_step
    * Condadependencies file: conda_dependencies_fullmodel.yml, including onnx and onnxruntime
    * Compute Target: 1 cpu cluster
    * Data sets: entire
    * Framework: PyTorch
