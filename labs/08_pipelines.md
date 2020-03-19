## Lab 8: Pipelines ##
In this lab, we are going to create a pipeline to execute the two steps of:
- hyperparameter tuning
- train full model
We are going to create a pipeline for these steps, because the input of the full model script is dependent on he ouput of the hyperparameter tuning script. With pipelines, it is very easy to orchastrate this.
Moreover, later we are going to implement tools for model monitoring. If our model is drifting, we want to retrain our model. Retraining the model consistst of the two steps mentioned above. In order to automate this, pipelines is a very good solution, 

## What are Azure ML pipelines?
An Azure ML pipeline performs a complete logical workflow with an ordered sequence of steps. Each step is a discrete processing action. Pipelines run in the context of an Azure Machine Learning Experiment.

In the early stages of an ML project, it's fine to have a single Jupyter notebook or Python script that does all the work of Azure workspace and resource configuration, data preparation, run configuration, training, and validation. But just as functions and classes quickly become preferable to a single imperative block of code, ML workflows quickly become preferable to a monolithic notebook or script.

By modularizing ML tasks, pipelines support the Computer Science imperative that a component should "do (only) one thing well." Modularity is clearly vital to project success when programming in teams, but even when working alone, even a small ML project involves separate tasks, each with a good amount of complexity. Tasks include: workspace configuration and data access, data preparation, model definition and configuration, and deployment. While the outputs of one or more tasks form the inputs to another, the exact implementation details of any one task are, at best, irrelevant distractions in the next. At worst, the computational state of one task can cause a bug in another.

![alt text](https://docs.microsoft.com/en-us/azure/machine-learning/media/concept-ml-pipelines/pipeline-flow.png)

## Creat Pipelinestep for hyperparametertuning
First, we are going to implement the pipeline step for hyperparameter tuning

1. Open the file `code\pipelines\pipeline_retrain.py` and load the required packages and configure the workspace
    ```
    from azureml.core import Workspace, Experiment, Datastore
    from azureml.core.runconfig import MpiConfiguration
    from azureml.core.authentication import AzureCliAuthentication
    from azureml.core.dataset import Dataset

    from azureml.train.dnn import PyTorch
    from azureml.train.hyperdrive.parameter_expressions import uniform, choice
    from azureml.train.hyperdrive import (
        BayesianParameterSampling,
        HyperDriveConfig, PrimaryMetricGoal)

    from azureml.pipeline.steps import HyperDriveStep
    from azureml.pipeline.core import Pipeline, PipelineData
    import os

    workspace = Workspace.from_config(auth=AzureCliAuthentication())
    ```
2.  Retrieve datastore/datasets
    ```
    # retrieve datastore
    datastore_name = 'workspaceblobstore'
    datastore = Datastore.get(workspace, datastore_name)

    # retrieve datasets used for training
    subset_dataset_train = Dataset.get_by_name(workspace,
                                               name='newsgroups_subset_train')
    subset_dataset_test = Dataset.get_by_name(workspace,
                                              name='newsgroups_subset_test')
    ```
3. Create output dataset for run metrics JSON file
The HyperDriveRun will produce a JSON file with the metrics from the run in it. We can use this file to substract the best run and its corresponding hyperparameters. For us to use the file in a later stage, we will store the JSON file in our storage account. In order to do so, we need to creat a data pipeline to our Datastore and sale the ouput in a folder named 'metrics_output':
    ```
    metrics_output_name = 'metrics_output'
    metrics_data = PipelineData(name='metrics_data',
                                datastore=datastore,
                                pipeline_output_name=metrics_output_name)
    ```

4. Define the compute target
    For every step in th pipeline, we can use different compute. As we did it the previous labs, we used two different kinds of compute for hyperparameter tuning and for training the full model. In this pipeline step we are going to use the compute we previously created when submitting the hyperparameter tuning file:
    ```
    compute_target_hyper = workspace.compute_targets["hypercomputegpu"]
    ```

5.  Define Run Configuration
    Next, we need to define our estimator. This is the same estimator we used in the previous labs:
    ```
    estimator = PyTorch(
        entry_script='train.py',
        source_directory=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '..',
            'modeling'
        ),
        compute_target=compute_target_hyper,
        distributed_training=MpiConfiguration(),
        framework_version='1.4',
        use_gpu=True,
        pip_packages=[
            'numpy==1.15.4',
            'pandas==0.23.4',
            'scikit-learn==0.20.1',
            'scipy==1.0.0',
            'matplotlib==3.0.2',
            'utils==0.9.0',
        ],
        inputs=[
            subset_dataset_train.as_named_input('subset_train'),
            subset_dataset_train.as_named_input('subset_test')
        ]
    )

6. Set the parameter search grid
    Last, we need to specify the search grid. This is again the same grid as that we have used in the previous labs.
    ```
    param_sampling = BayesianParameterSampling({
        "learning_rate": uniform(0.05, 0.1),
        "num_epochs": choice(5, 10, 15),
        "batch_size": choice(150, 200),
        "hidden_size": choice(50, 100)
    })
    ```

7. Define the pipeline step
    We are now going to define the pipeline step. Azure ML Pipeline steps can be configured together to construct a Pipeline, which represents a shareable and reusable Azure Machine Learning workflow. Each step of a pipeline can be configured to allow reuse of its previous run results if the step contents (scripts and dependencies) as well as inputs and parameters remain unchanged. To make it even more easy, Azure ML had a HyperDriveStep This creates an Azure ML Pipeline step to run hyperparameter tunning for Machine Learning model training. For more info check: https://docs.microsoft.com/en-us/python/api/azureml-pipeline-steps/azureml.pipeline.steps.hyper_drive_step.hyperdrivestep?view=azure-ml-py
    ```
    hypertuning = HyperDriveStep(
                name='hypertrain',
                hyperdrive_config=HyperDriveConfig(
                    estimator=estimator,
                    hyperparameter_sampling=param_sampling,
                    policy=None,
                    primary_metric_name="accuracy",
                    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                    max_total_runs=80,
                    max_concurrent_runs=None
                ),
                estimator_entry_script_arguments=[],
                inputs=[
                        subset_dataset_train.as_named_input('subset_train'),
                        subset_dataset_test.as_named_input('subset_test')
                        ],
                outputs=[],
                metrics_output=metrics_data,
                allow_reuse=True,
                version=None
    )
    ```
8. Attach step to the pipeline
    Every step that we create, we can easily attach to the pipeline. By attaching steps to the pipeline, we can create a logical pipeline that will excecute specific task in a specific order and use outputs of one step as input in the next step.
    `pipeline = Pipeline(workspace=workspace, steps=hypertuning)`

9. Submit the pipeline
    Submitting the pipeline is similair as to running an experiment.
    ```
    # Define the experiment
    experiment = Experiment(workspace, 're-train')

    # Run the experiment
    pipeline_run = experiment.submit(pipeline)
    pipeline_run.wait_for_completion()
    ```
10. Go to the portal and inspect the results.

## Creat Pipelinestep for full model
We are now going to define the pipeline step. In this step we are going to make use of the PythonScriptStep. This is the stantard step of executing a Python script in a pipeline. We need to take the following steps in this part of the tuturial:
    - Retrive the entire data from the Datasets
    - Define the compute target
    - Define the conda dependencies
    - Define the Run COnfig
    - Define the Pipeline step
    - Add the step to the pipeline
    - Run the script

1.  Retrieve datastore/datasets
    ```
    dataset_train = Dataset.get_by_name(workspace,
                                        name='newsgroups_train')
    dataset_test = Dataset.get_by_name(workspace,
                                        name='newsgroups_test')
    ```

4. Define the compute target
    For every step in th pipeline, we can use different compute. As we did it the previous labs, we used two different kinds of compute for hyperparameter tuning and for training the full model. In this pipeline step we are going to use the compute we previously created when submitting the full model train file:
    `compute_target_fullmodel = workspace.compute_targets["fullcomputegpu"]`

5.  Define Cond adependencies:
    Next, we need to define our estimator. This is the same estimator we used in the previous labs:
    ```
    estimator = PyTorch(
        entry_script='train.py',
        source_directory=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '..',
            'modeling'
        ),
        compute_target=compute_target_hyper,
        distributed_training=MpiConfiguration(),
        framework_version='1.4',
        use_gpu=True,
        pip_packages=[
            'numpy==1.15.4',
            'pandas==0.23.4',
            'scikit-learn==0.20.1',
            'scipy==1.0.0',
            'matplotlib==3.0.2',
            'utils==0.9.0',
        ],
        inputs=[
            subset_dataset_train.as_named_input('subset_train'),
            subset_dataset_train.as_named_input('subset_test')
        ]
    )

6. Set the parameter search grid
    Last, we need to specify the search grid. This is again the same grid as that we have used in the previous labs.
    ```
    param_sampling = BayesianParameterSampling({
        "learning_rate": uniform(0.05, 0.1),
        "num_epochs": choice(5, 10, 15),
        "batch_size": choice(150, 200),
        "hidden_size": choice(50, 100)
    })
    ```

7. Define the pipeline step
    We are now going to define the pipeline step. In this step we are going to make use of the PythonScriptStep. This is the stantard step of executing a Python script in a pipeline.
    ```
    hypertuning = HyperDriveStep(
                name='hypertrain',
                hyperdrive_config=HyperDriveConfig(
                    estimator=estimator,
                    hyperparameter_sampling=param_sampling,
                    policy=None,
                    primary_metric_name="accuracy",
                    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                    max_total_runs=80,
                    max_concurrent_runs=None
                ),
                estimator_entry_script_arguments=[],
                inputs=[
                        subset_dataset_train.as_named_input('subset_train'),
                        subset_dataset_test.as_named_input('subset_test')
                        ],
                outputs=[],
                metrics_output=metrics_data,
                allow_reuse=True,
                version=None
    )
    ```
8. Attach step to the pipeline
    Every step that we create, we can easily attach to the pipeline. By attaching steps to the pipeline, we can create a logical pipeline that will excecute specific task in a specific order and use outputs of one step as input in the next step.
    `pipeline = Pipeline(workspace=workspace, steps=hypertuning)`

9. Submit the pipeline
    Submitting the pipeline is similair as to running an experiment.
    ```
    # Define the experiment
    experiment = Experiment(workspace, 're-train')

    # Run the experiment
    pipeline_run = experiment.submit(pipeline)
    pipeline_run.wait_for_completion()
    ```
10. Go to the portal and inspect the results.

