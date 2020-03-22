## Lab 4: Hyperparamter tuning ##
 In this lab we are going to build a Deep Neural Network using Pytorch. We are going to use the capabilities of Azure Machine Leaning to perform distributed training on GPU's. This we will see in future labs. In this lab, we are going to see how we can levarge Azure ML to do hyeperparameter tuning of our neural network. 

Efficiently tune hyperparameters for your model using Azure Machine Learning. Hyperparameter tuning includes the following steps:
* Define the parameter search space
* Specify a primary metric to optimize
* Specify early termination criteria for poorly performing runs
* Allocate resources for hyperparameter tuning
* Launch an experiment with the above configuration
* Visualize the training runs
* Select the best performing configuration for your model

# Pre-requirements #
1. Completed lab [03_childrun](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/labs/03_childrun.md)
2. Familiarize yourself with the concept of [Deep Learning on Azure ML]
3. Read the documentation on [How to tune hyperparameters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
4. Read the documentation on [Train Pytorch deep learning models at scale](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch)


## What are hyperparameters?
Hyperparameters are adjustable parameters you choose to train a model that govern the training process itself. For example, to train a deep neural network, you decide the number of hidden layers in the network and the number of nodes in each layer prior to training the model. These values usually stay constant during the training process.


## Bayesian sampling
Bayesian sampling is based on the Bayesian optimization algorithm and makes intelligent choices on the hyperparameter values to sample next. It picks the sample based on how the previous samples performed, such that the new sample improves the reported primary metric.

When you use Bayesian sampling, the number of concurrent runs has an impact on the effectiveness of the tuning process. Typically, a smaller number of concurrent runs can lead to better sampling convergence, since the smaller degree of parallelism increases the number of runs that benefit from previously completed runs.

Bayesian sampling only supports choice, uniform, and quniform distributions over the search space.

## Distributed training
The PyTorch estimator also supports distributed training across CPU and GPU clusters. You can easily run distributed PyTorch jobs and Azure Machine Learning will manage the orchestration for you.

Horovod
Horovod is an open-source, all reduce framework for distributed training developed by Uber. It offers an easy path to distributed GPU PyTorch jobs.

To use Horovod, specify an MpiConfiguration object for the distributed_training parameter in the PyTorch constructor. This parameter ensures that Horovod library is installed for you to use in your training script.

# Understand the non-azure / open source ml model code #
In this lab, we are going to use the same dataset from from https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html. We will do the same dataprocessing.

Since the models tested in our explorative fase were not good enough for our use case, we are going to make use of Deep Learning. To build our Deep Neural Network, we are going to make use of Pytorch. Pytorch is an open source machine learning framework that accelerates the path from research prototyping to production deployment. For more infor seee https://pytorch.org/. 

In this tututial, we are going to build a Deep Neural Network with two layers, following this structure:

```python
class OurNet(nn.Module):

     def __init__(self, input_size, hidden_size, num_classes):
        super(OurNet, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

     def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out
```

For more information on deep learning check: https://docs.microsoft.com/en-us/azure/machine-learning/concept-deep-learning-vs-machine-learning. 


# Run the training locally #
Just to check, we are now going to train the script locally without using Azure ML. 
1. Execute the script `code/modeling/train.py`

#  Run the code via Azure ML #
Running the code via Azure ML, we need to excecute two steps. First, we need to refactor the training script. Secondly, we need to create a submit_train file to excecute the train file.

## Part 1: Refactor the code to capture run metrics in train.py 

1. Get the run context
    
    ```python
    from azureml.core import Run
    run_logger = Run.get_context()
    ```

2. Log the metric in the run
    In this example, we are going to log multiple metrics. First the accuracy, as we have done in the preious labs, but also all the hyperparameters from that run:

    ```python
    run_logger.log("accuracy", float(accuracy))
    run_logger.log("learning_rate", learning_rate)
    run_logger.log("num_epochs", num_epochs)
    run_logger.log("batch_size", batch_size)
    run_logger.log("hidden_size", hidden_size)
    ```

    Secondly, we are going to log the graphs that display the Accuracy and loss per epoch that we run with the `log_image()` syntax. Use this method to log an image file or a matplotlib plot to the run. These images will be visible and comparable in the run record.
    Replace:
    ```python
    plt.show()
    ```
    with:
    ```python
    run_logger.log_image("Loss grapgh", plot=plt)
    ```
    and:
    ```python
    run_logger.log_image("Accuracy", plot=plt)
    ```

3. upload the .pkl file to the output folder
    
    ```python
    from sklearn.externals import joblib
    OUTPUTSFOLDER = "outputs"

    # save .pkl file
    model_name = "model" + ".pkl"
    filename = os.path.join(OUTPUTSFOLDER, model_name)
    joblib.dump(value=outputs, filename=filename)
    run_logger.upload_file(name=model_name, path_or_stream=filename)
    ```

4. close the run
    ```python
    run.complete()
    ```

5. add parameters to the script to `op = OptionParser()`
    Now that we have logged all the metrics, we want to enable the script to take hyperparameters as arguments, so we can submit those arguments later to the scrip when we are running the HyperDrive. In this lab, we are going to tune 4 hyperparameters from a neural network, including the learning rate, number of epochs, batch size and the number of nodes in the hidden layers. In order to enable our script to take these parameters as arguments, we need to add them to our parser:

    ```python
    op.add_option("--learning_rate",
                  type=float, default=0.01)
    op.add_option("--num_epochs",
                  type=int, default=2)
    op.add_option("--batch_size",
                  type=int,
                  default=150)
    op.add_option("--hidden_size",
                  type=int,
                  default=100)
    ```

6. Replace the hard-coded parameters by the arguments
    We are now going to replace the hard-coded parameters by the argument we just specified. We are going to replace the following piece of code:

    ```python
    # Parameters
    learning_rate = 0.01
    num_epochs = 20
    batch_size = 300

    # Network Parameters
    hidden_size = 100      # 1st layer and 2nd layer number of features
    ```
    By the following:
    ```python
    hyperparameters = {
        "learning_rate": opts.learning_rate,
        "num_epochs": opts.num_epochs,
        "batch_size": opts.batch_size,
        "hidden_size": opts.hidden_size,
    }

    # Select the training hyperparameters.
    learning_rate = hyperparameters["learning_rate"]
    num_epochs = hyperparameters["num_epochs"]
    batch_size = hyperparameters["batch_size"]
    hidden_size = hyperparameters["hidden_size"]
    ```

7. Execute the refactored script `code/explore/train.py`
    As an output you should get the following:

    ```python  
    Loading 20 newsgroups dataset for categories:
    data loaded
    Attempted to log scalar metric accuracy:
    0.7915742793791575
    Attempted to track file model.pkl at outputs\model.pkl
    ```
    
    As we have seen in previous labs, we need to create a submit file to submit the run to Azure ML. We will do this in the next part.

Note: The completed code can be found [here](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/code_labs/modeling/train.py)

## ALter the deeptrain_submit.py file

1. Load required Azureml libraries
    ```
    from azureml.train.hyperdrive import (
        BayesianParameterSampling,
        HyperDriveConfig, PrimaryMetricGoal)
    from azureml.core import Workspace, Experiment
    from azureml.core.runconfig import MpiConfiguration
    from azureml.train.estimator import Estimator
    import os
    from azureml.train.hyperdrive.parameter_expressions import uniform, choice
    from azureml.core.authentication import AzureCliAuthentication
    from azureml.train.dnn import PyTorch

    ```

2. Load Azure ML workspace form config file
    ```
    # load Azure ML workspace
    workspace = Workspace.from_config(auth=AzureCliAuthentication())
    ```

3. Create an extimator to define the run configuration
    Note here that we are using a special PyTorch Estimator. The PyTorch estimator also supports distributed training across CPU and GPU clusters. You can easily run distributed PyTorch jobs and Azure Machine Learning will manage the orchestration for you. When submitting a training job, Azure ML runs your script in a conda environment within a Docker container. The PyTorch containers have the [following](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.dnn.pytorch?view=azure-ml-py) dependencies installed.

    We are setting the `distributed_training` to `MpiConfiguration()`. This way we allow for distributed training with Mpi as backend. Message Passing Interface (MPI) is a standardized and portable message-passing system developed for distributed and parallel computing. The `framework_version` is the Pytorch framework version that we are using. In our case this is 1.4. 

    Note: We are setting the compute to `'local'`. It is not possible to excecute the Pytorch estimator locally, but in the next lab we will show how to attach remote compute.

    ```python
    # Define Run Configuration
    estimator = PyTorch(
        entry_script='train.py',
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        compute_target='local',
        distributed_training=MpiConfiguration(),
        framework_version='1.4',
        use_gpu=False,
        conda_dependencies_file=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../',
            'conda_dependencies.yml'
        )
    )

    ```

4. Set search parameters
    In this step we are going to define the search space for hyperparameter tuning. These number are random. However, it is advised to use a search space based on prior knowledge or acedemic research in order to optimize the hyperparameter tuning.

    ```python
    # Set parameters for search
    param_sampling = BayesianParameterSampling({
        "learning_rate": uniform(0.05, 0.1),
        "num_epochs": choice(5, 10, 15),
        "batch_size": choice(150, 200),
        "hidden_size": choice(50, 100)
    })
    ```

5. Define the Hyperdrive run configration
    Because we are running an HyperDrive for parameter tuning, we need to define the `HyperDriveConfig`. HyperDrive configuration includes information about hyperparameter space sampling, termination policy, primary metric, resume from configuration, estimator, and the compute target to execute the experiment runs on.

    ```python
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

4. Define the ML experiment

    ```python
    # Define the ML experiment
    experiment = Experiment(workspace, "newsgroups_train_hypertune")
    ```

5. Submit the experiment

    ```python
   # Submit the experiment
    hyperdrive_run = experiment.submit(hyperdrive_run_config)
    hyperdrive_run.wait_for_completion()
    ```

6. Run the script `code\explore\deeptrain_submit.py`
You will get an error message.
This error message occurs, because we tried to run our script locally. This is not possible for HyperDrive. If we want no run this, we need to use Azure ML compute. In the next tuturial we are goin to create remote compute and how to submit a run on remote compute.

Note: The completed code can be found [here](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/code_labs/modeling/train_hyper_submit.py)
