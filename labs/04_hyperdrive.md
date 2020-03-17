## Lab 2: running experiments ##

# Understand the non-azure / open source ml model code #

# Run the training locally #
We are now going to train the script locally without using Azure ML. 
Execute the script `code/explore/traindeep.py`

#  Run the code via Azure ML #
We are now going to run our code via Azure ML. 

1. Read Experiment Tracking documentation

2. Read How to Mange a Run documentation

Running the code via Azure ML, we need to excecute two steps. First, we need to refactor the training script. Secondly, we need to create a submit_train file to excecute the train file.

## Refactor the code to capture run metrics in train.py 

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

## ALter the train_submit.py file

1. Load required Azureml libraries
    ```
    from azureml.train.hyperdrive import (
    RandomParameterSampling,
    HyperDriveConfig, PrimaryMetricGoal)
    from azureml.core import Workspace, Experiment
    from azureml.train.estimator import Estimator
    import pandas as pd
    import os
    from random import choice
    from azureml.core.authentication import AzureCliAuthentication
    ```

2. Load Azure ML workspace form config file
    ```
    # load Azure ML workspace
    workspace = Workspace.from_config(auth=AzureCliAuthentication())
    ```

3. Create an extimator to define the run configuration
    ```
    # Define Run Configuration
    estimator = Estimator(
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
            'torch==1.4.0'
        ]
    )

    ```

4. Set search parameters
    ```
    # Set parameters for search

    param_sampling = RandomParameterSampling({
        "learning_rate": choice([0.01, 0.02]),
        "num_epochs": choice([5, 10]),
        "batch_size": choice([150, 200]),
        "hidden_size": choice([100, 150]),
        }
    )
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
        max_total_runs=2,
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

6. Select the best model and the corresponding parameters
    ```
    # Select the best run from all submitted
    best_run = hyperdrive_run.get_best_run_by_primary_metric()
    best_run_metrics = best_run.get_metrics()

    # Log the best run's performance to the parent run
    hyperdrive_run.log("Accuracy", best_run_metrics['accuracy'])
    parameter_values = best_run.get_details()['runDefinition']['arguments']

    # Best set of parameters found
    best_parameters = dict(zip(parameter_values[::2], parameter_values[1::2]))
    best_model_parameters = best_parameters.copy()
    ```

7. Go to the portal to inspect the run history

Note: the correct code is already available in codeazureml. In here, all ready to use code is available for the entire workshop.
