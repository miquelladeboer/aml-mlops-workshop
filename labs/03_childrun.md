## Lab 3: creating child runs ##

#  Run the code via Azure ML #
In tuturial 2, we expored the Random Forest classifier in order to classify news articles. The performance of the algorithm was not as good as expected, so we want to extend our explorative fase.

In this tuturial, we are going to benchmark 15 different algoritms in order to compare performance across algorithms. We are going to log the accuracy metric for every algorithm in Azure ML. Next, we are going to select the best model and register the model to Azure ML.

Comparing different algorithms is possible is different ways. We could submit a new experiment for every algorithm that we try. However, Azure ML offers a better, easier way to manage the exploration of multiple models. This concept is called child runs.  We are going to make use of these child runs. The expiriment will perform a parent run that is going to execute `train15.py`. Within `train.py15` we are going to create child runs. For every of the 15 algoritms that we have we want to create a sub run and log the metrics seprately. Whihin the child run we are going to log the performane and the model .pkl files. This way we can easily track and compare our experiment in Azure ML.

# Understand the non-azure / open source ml model code #
The training script for running the 15 different algorithms is the same as in tuturial 2, but instead of running only Random Forest Clasfier as below:
```
clf = RandomForestClassifier()
name =  "Random forest"

benchmark(clf, name)
```
We are now going to run 15 algorithms as follows:
```
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

The training script is already prepared for you and you can find it in `\code\explore\train15.py`

# Run the training locally #
We are now going to train the script locally without using Azure ML. 
Execute the script `code/explore/train15.py`

#  Run the code via Azure ML #
We are now going to run our code via Azure ML. 

1. Read Experiment Tracking documentation

2. Read How to Mange a Run documentation

Running the code via Azure ML, we need to excecute two steps. First, we need to refactor the training script. Secondly, we need to create a submit_train file to excecute the train file.

## Refactor the code to capture run metrics in train15.py 
In the main file:

1. Get the run context
    ```
    from azureml.core import Run
    run = Run.get_context()
    ```
For each algorithm in the loop:

2. Create a child run
    ```
    # create a child run for Azure ML logging
    child_run = run.child_run(name=name)
    ```
3. Log the metric in the run

    `child_run.log("accuracy", float(score))`

4. upload the .pkl file to the output folder
    
    ```
    # write model artifact
    model_name = "model" + str(name) + ".pkl"
    filename = "outputs/" + model_name

    # upload model artifact with child run
    child_run.upload_file(
        name=model_name,
        path_or_stream=filename
    )
    ```

5. close the child run

    `child_run.complete()`

6. Execute the refactored script `code/explore/train15.py`
For every algorithm, as an output you should get something similar to the the following:
```
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

## ALter the train_submit.py file

1. Alter the estimator to the right script from `train.py` to `train15.py`
    ```
    # Define Run Configuration
    est = Estimator(
    entry_script='train15.py',
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    compute_target='local',
    conda_packages=[
        'pip==20.0.2'
    ],
    pip_packages=[
        'numpy==1.15.4',
        'pandas==0.23.4',
        'scikit-learn==0.20.1',
        'scipy==1.0.0',
        'matplotlib==3.0.2',
        'utils==0.9.0'
    ],
    use_docker=False
    )
    ```

2. Alter the ML experiment name `newsgroups_train` to `newsgroups_train15`
    ```
    # Define the ML experiment
    experiment = Experiment(workspace, "newsgroups_train15")
    ```

3. Submit the experimen

4. Go to the portal to inspect the run history

Note: the correct code is already available in codeazureml. In here, all ready to use code is available for the entire workshop.


