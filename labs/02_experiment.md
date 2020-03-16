## Lab 2: running experiments ##

# Understand the non-azure / open source ml model code #
We first start with understanding the training script. The training script is an open source ML model code from https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html.  The dataset used in this example is the 20 newsgroups dataset. It will be automatically downloaded, then cached. The newsgroup datasets contains text documents that are classified into 20 categories.

Open the train.py document to inspect the code.
The first step in the code is to load the dataset from the 20 newsgroup dataset. In this example we are only going to use a subset of the categories. Please state the catogories we are going to use:

...

The second step is to extract the features from the text. We do this with a sparse vecorizer. We also clean the data a bit. What is the operation that we do on the data to clean the text?

...

After we have reshaped our data and made sure the feature names are in the right place, we are going to define the algorithm to fit the model. This step is defining the benchmark. We fit the data and make predictions on the test set. To validate out model we need a metric to score the model. There are many metrics we can use. Define in the code the metric that you want to use to validate your model and make sure the print statement will output your metric. (Note: you can define multiple scores if you want. If so, make sure to return these scores.)

...


The last step is to define tha algoritms that we want to fit over our data. In this example we are using 1 classification algoritm to fit the data. We keep track of the metrics of all algoritms, so we can compare the performance and pick the model. Look at the code and whrite down the algoritm that we are going to test.

...

# Run the training locally #
We are now going to train the script locally without using Azure ML. 
Execute the script `code/explore/train.py`

#  Run the code via Azure ML #
We are now going to run our code via Azure ML. 

1. Read Experiment Tracking documentation

2. Read How to Mange a Run documentation

Running the code via Azure ML, we need to excecute two steps. First, we need to refactor the training script. Secondly, we need to create a submit_train file to excecute the train file.

## Refactor the code to capture run metrics in train.py 

1. Get the run context
    ```
    from azureml.core import Run
    run = Run.get_context()
    ```

2. Log the metric in the run

    `run.log("accuracy", float(score))`

3. upload the .pkl file to the output folder
    
    ```
    # write model artifact

    model_name = "model" + str(name) + ".pkl"
    filename = "outputs/" + model_name
    run.upload_file(name=model_name, path_or_stream=filename)
    ```

4. close the run

    `run.complete()`

5. Execute the refactored script `code/explore/train.py`
As an output you should get the following:
```
Attempted to log scalar metric accuracy:
0.7834441980783444
Attempted to track file modelRandom forest.pkl at outputs/modelRandom forest.pkl
Accuracy  0.783
```

## ALter the train_submit.py file

1. Load required Azureml libraries
    ```
    from azureml.core import Workspace, Experiment
    from azureml.train.estimator import Estimator
    ```

2. Load Azure ML workspace form config file
    ```
    # load Azure ML workspace
    workspace = Workspace.from_config(auth=AzureCliAuthentication())
    ```

3. Create an extimator to define the run configuration
    ```
    # Define Run Configuration
    est = Estimator(
    entry_script='train.py',
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

4. Define the ML experiment
    ```
    # Define the ML experiment
    experiment = Experiment(workspace, "newsgroups_train")
    ```

5. Submit the experiment
    ```
    # Submit experiment run, if compute is idle, this may take some time')
    run = experiment.submit(est)

    # wait for run completion of the run, while showing the logs
    run.wait_for_completion(show_output=True)
    ```


6. Go to the portal to inspect the run history

Note: the correct code is already available in codeazureml. In here, all ready to use code is available for the entire workshop.

