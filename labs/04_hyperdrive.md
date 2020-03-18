## Lab 4: Hyperparamter tuning ##
In this lab, we are going to see how we can levarge Azure ML to do hyeperparameter tuning. 
After we have performed hypertuning, we will use the best parameters to train the full model.

Efficiently tune hyperparameters for your model using Azure Machine Learning. Hyperparameter tuning includes the following steps:

1. Define the parameter search space
2. Specify a primary metric to optimize
3. Specify early termination criteria for poorly performing runs
4. Allocate resources for hyperparameter tuning
5. Launch an experiment with the above configuration
6. Visualize the training runs
7. Select the best performing configuration for your model

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
```
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
We are now going to train the script locally without using Azure ML. 
Execute the script `code/explore/traindeep.py`

#  Run the HyperDrive via Azure ML #
We are now going to run our code via Azure ML. 

1. Read Experiment Tracking documentation
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters

Running the code via Azure ML, we need to excecute two steps. First, we need to refactor the training script. Secondly, we need to create a submit_train file to excecute the train file.

## Refactor the code to capture run metrics in deeptrain.py 

1. Get the run context
    ```
    from azureml.core import Run
    run_logger = Run.get_context()
    ```

2. Log the metric in the run

    `run_logger.log("accuracy", float(accuracy))`

3. upload the .pkl file to the output folder
    
    ```
    from sklearn.externals import joblib
    OUTPUTSFOLDER = "outputs"
    # save .pkl file
    model_name = "model" + ".pkl"
    filename = os.path.join(OUTPUTSFOLDER, model_name)
    joblib.dump(value=outputs, filename=filename)
    run_logger.upload_file(name=model_name, path_or_stream=filename)
    ```

4. close the run

    `run.complete()`

5. add parameters to the script to `op = OptionParser()`
```
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

6. Replace the hard-coded paramters by the arguments
Replace the following code:
```
# Parameters
learning_rate = 0.01
num_epochs = 20
batch_size = 300

# Network Parameters
hidden_size = 100      # 1st layer and 2nd layer number of features
```
By the following:
```
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

7. Execute the refactored script `code/explore/deeptrain.py`
As an output you should get the following:
    ```
    Loading 20 newsgroups dataset for categories:
    data loaded
    Attempted to log scalar metric accuracy:
    0.7915742793791575
    Attempted to track file model.pkl at outputs\model.pkl
    ```

## ALter the deeptrain_submit.py file

1. Load required Azureml libraries
    ```
    from azureml.train.hyperdrive import (
        BayesianParameterSampling,
        HyperDriveConfig, PrimaryMetricGoal)
    from azureml.core import Workspace, Experiment
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
    Note here that we are using a special PyTorch Estimator. The PyTorch estimator also supports distributed training across CPU and GPU clusters. You can easily run distributed PyTorch jobs and Azure Machine Learning will manage the orchestration for you.
    ```
    # Define Run Configuration
    estimator = PyTorch(
        entry_script='traindeep.py',
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        compute_target='local',
        pip_packages=[
            'numpy==1.15.4',
            'pandas==0.23.4',
            'scikit-learn==0.20.1',
            'scipy==1.0.0',
            'matplotlib==3.0.2',
            'utils==0.9.0',
        ]
    )

    ```

4. Set search parameters
    ```
    # Set parameters for search
    param_sampling = BayesianParameterSampling({
        "learning_rate": uniform(0.05, 0.1),
        "num_epochs": choice(5, 10, 15),
        "batch_size": choice(150, 200),
        "hidden_size": choice(50, 100)
    })
    ```

5. Define the Hyperdrice run configration
    ```
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
    ```
    # Define the ML experiment
    experiment = Experiment(workspace, "newsgroups_train_hypertune")
    ```

5. Submit the experiment
    ```
   # Submit the experiment
    hyperdrive_run = experiment.submit(hyperdrive_run_config)
    hyperdrive_run.wait_for_completion()
    ```

6. Run the script `code\explore\deeptrain_submit.py`
You will get an error message.
This error message occurs, because we tried to run our script locally. This is not possible for HyperDrive. If we want no run this, we need to use Azure ML compute. In the next tuturial we are goin to create remote compute and how to submit a run on remote compute, but firt we need to create a script for the full model.

Note: the correct code is already available in codeazureml. In here, all ready to use code is available for the entire workshop.

#  Run the full model via Azure ML #
We are now going to use the parameters from the best model of the Hyperdrive to submit in the full model. Be awere that we are not able yet, to run the HyperDrive and retrieve the right parameters, but we are going to set up the script already, so we can use it once we have the right compute. We are going to use the same train script `traindeep.py` and only going to create a new submit file, were we pass the best parameter through the script, instead of the hypertune search space.

1. Open the file `traindeep_fullmodel_submit.py`
