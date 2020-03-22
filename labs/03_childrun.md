## Lab 3: creating child runs ##
In tuturial 2, we expored the Random Forest classifier in order to classify news articles. The performance of the algorithm was not as good as expected, so we want to extend our explorative fase.

In this tuturial, we are going to benchmark 15 different algoritms in order to compare performance across algorithms. We are going to log the accuracy metric for every algorithm in Azure ML. Next, we are going to select the best model and register the model to Azure ML. In this tuturial you will learn:
* How to benchmark different models
* How to create child runs
* How to log metrics to child runs
* Select the best model from a run with chilld runs

# Pre-requirements #
1. Completed lab [02_experiments](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/labs/02_experiments.md)


# Understand the non-azure / open source ml model code #
Comparing different algorithms is possible is different ways. We could submit a new experiment for every algorithm that we try. However, Azure ML offers a better, easier way to manage the exploration of multiple models. This concept is called child runs.  We are going to make use of these child runs. The expiriment will perform a parent run that is going to execute `explore/code/train_15models.py`. Within this file we are going to create child runs. For every of the 15 algoritms that we have we want to create a sub run and log the metrics seprately. Whihin the child run we are going to log the performane and the model .pkl files. This way we can easily track and compare our experiment in Azure ML. The training script for running the 15 different algorithms is the same as in tuturial 2, but instead of running only Random Forest Clasfier as below:

```python
clf = RandomForestClassifier()
name =  "Random forest"

benchmark(clf, name)
```

We are now going to run 15 algorithms as follows:

```python
# Run benchmark and collect results with multiple classifiers
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(), "Random forest")):
    # run benchmarking function for each
    benchmark(clf, name)


# Run with different regularization techniques
for penalty in ["l2", "l1"]:
    # Train Liblinear model
    name = penalty + "LinearSVC"
    benchmark(
        clf=LinearSVC(
            penalty=penalty,
            dual=False,
            tol=1e-3
        ),
        name=penalty + "LinearSVC"
    )

    # Train SGD model
    benchmark(
        SGDClassifier(
            alpha=.0001,
            max_iter=50,
            penalty=penalty
        ),
        name=penalty + "SGDClassifier"
    )

# Train SGD with Elastic Net penalty
benchmark(
    SGDClassifier(
        alpha=.0001,
        max_iter=50,
        penalty="elasticnet"
    ),
    name="Elastic-Net penalty"
)

# Train NearestCentroid without threshold
benchmark(
    NearestCentroid(),
    name="NearestCentroid (aka Rocchio classifier)"
)

# Train sparse Naive Bayes classifiers
benchmark(
    MultinomialNB(alpha=.01),
    name="Naive Bayes MultinomialNB"
)

benchmark(
    BernoulliNB(alpha=.01),
    name="Naive Bayes BernoulliNB"
)

benchmark(
    ComplementNB(alpha=.1),
    name="Naive Bayes ComplementNB"
)

# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
benchmark(
    Pipeline([
        ('feature_selection',
            SelectFromModel(
                LinearSVC(
                    penalty="l1",
                    dual=False,
                    tol=1e-3
                )
            )),
        ('classification',
            LinearSVC(penalty="l2"))
        ]
    ),
    name="LinearSVC with L1-based feature selection"
)
```

The training script is already prepared for you and you can find it in `\code\explore\train_15models.py`

# Run the training locally #
Just to check, we are now going to train the script locally without using Azure ML. 
1. Execute the script `code/explore/train_15models.py`

#  Run the code via Azure ML #
Running the code via Azure ML, we need to excecute two steps. First, we need to refactor the training script. Secondly, we need to create a submit_train file to excecute the train file.

## Part 1: Refactor the code to capture run metrics in code/explore/train_15models.py

1. Get the run context

    ```python
    from azureml.core import Run
    run = Run.get_context()
    ```

2. Create a child run
    For each algorithm in the loop, we are going to create a child run. This way we can store the metrics of the models in different child runs, so we can easily compare the models and select the best model for future training or maybe even model deployment for production. We are going to use the  `child_run()` statement for this. This is used to isolate part of a run into a subsection. This can be done for identifiable "parts" of a run that are interesting to separate, or to capture independent metrics across an interation of a subprocess. If an output directory is set for the child run, the contents of that directory will be uploaded to the child run record when the child is completed. 

    ```python
    # create a child run for Azure ML logging
    child_run = run.child_run(name=name)
    ```

3. Log the metric in the run
    Next, we are going to log the accuracy for every model in their own child run. This statement is the same as you would log your metrics to a parent run. 

    ```python
    child_run.log("accuracy", float(score))
    ```

4. upload the .pkl file to the output folder
    
    ```python
    The pkl file contains the model we have trained. We want to upload the model to the Azure Machine Learning Service. We need to create an outputs folder if one is not present yet where we can save the model at the top of the file.

    ```python
    import os

    # Define ouputs folder
    OUTPUTSFOLDER = "outputs"

    # create outputs folder if not exists
    if not os.path.exists(OUTPUTSFOLDER):
        os.makedirs(OUTPUTSFOLDER)
    ```

    Next, at the end of the file, once we have trainded our model, we want to save that model in the outputs folder we have just created, by the following code:

    ```python
    from sklearn.externals import joblib

    # save .pkl file
    model_name = "model" + ".pkl"
    filename = os.path.join(OUTPUTSFOLDER, model_name)

    # upload model artifact with child run
    child_run.upload_file(
        name=model_name,
        path_or_stream=filename
    )
    ```

5. close the child run
    ```python
    child_run.complete()
    ```

6. Execute the refactored script `code/explore/train15.py`
    For every algorithm, as an output you should get something similar to the the following:

    ```python
    Training run with algorithm
    Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
               fit_intercept=True, max_iter=50, n_iter_no_change=5, n_jobs=None,
               penalty=None, random_state=0, shuffle=True, tol=0.001,
               validation_fraction=0.1, verbose=0, warm_start=False)
    Attempted to log scalar metric accuracy:
    0.8876570583887657
    Attempted to track file modelPerceptron.pkl at outputs/modelPerceptron.pkl
    Accuracy  0.888
    ```

    As we have seen in the previous lab, the code is not submitting our logs to Azure ML. In order to do this, we need to submit the run as an experiment to Azure ML. Therefore we are going to create a submit file in the next part. 

Note: The completed code can be found [here](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/code_labs/explore/train_15models.py)

## Part 2: Create the train_15models_submit.py file
In this part, we are going to create the submit file.
1. Load required Azureml libraries

    ```python
    import os
    from azureml.core import Workspace, Experiment
    from azureml.train.estimator import Estimator
    from azureml.core.authentication import AzureCliAuthentication
    ```

2. Load Azure ML workspace form config file

    ```python
    # load Azure ML workspace
    workspace = Workspace.from_config(auth=AzureCliAuthentication())
    ```

3. Create an extimator to define the run configuration

    ```python
    # Define Run Configuration
    est = Estimator(
        entry_script='train_15models.py',
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        compute_target='local',
        conda_dependencies_file=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../',
            'conda_dependencies.yml'
        ),
        use_docker=False
    )
    ```

4. Define the ML experiment

    ```python
    # Define the ML experiment
    experiment = Experiment(workspace, "newsgroups_train_15models")
    ```

5. Submit the experiment
    We are now ready to submit the experiment:

    ```python
    # Submit experiment run, if compute is idle, this may take some time')
    run = experiment.submit(est)

    # wait for run completion of the run, while showing the logs
    run.wait_for_completion(show_output=True)
    ```

6. Get the best results from the run
    Print the best results from the run:

    ```python
    max_run_id = None
    max_accuracy = None

    for run in experiment.get_runs():
        run_metrics = run.get_metrics()
        run_details = run.get_details()
        # each logged metric becomes a key in this returned dict
        accuracy = run_metrics["accuracy"]
        run_id = run_details["runId"]

        if max_accuracy is None:
            max_accuracy = accuracy
            max_run_id = run_id
        else:
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_runid = run_id

    print("Best run_id: " + max_run_id)
    print("Best run_id acuuracy: " + str(max_accuracy))
    ```

7. Go to the portal and inspect the results

Note: The completed code can be found [here](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/code_labs/explore/train_15models_submit.py)
