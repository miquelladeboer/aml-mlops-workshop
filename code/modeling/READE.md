# Modeling
This folder contains scripts for modeling. This folder contains three files: a training script for training the model, a training submit script to submit the training to Azure Machine Learning and a script for model scoring, that we need when we deploy the model

*  `train.py`

    In this file we will do the model training. This file is set up to do 3 types of model trainig. In the parameters on the top of the script, we can change the types of models that we are training, as well as the data that we want to use for training. In this example we will see how we can easily switch from local to remote datasets and compute as well as switch between types of datasets, as we might want to train a heavy model in the experimentation phase only on a subsample of the data.

    In this file you can find models for:

    * random forest

        If we set the parameter of `--models` to `randomforest`, we train script will fit a random forest from sklearn over the data. We will log different metrics and plots to Azure Machine Learning inlcuding the confusion matrix, AUC curve and accuracy metrics. The output in Azure Machine learning will look similair to this. (Note that I am using old experience of the studio here as in my opion this give a better overview of my runs).
        ![An example of Random Forest](attributesrandomforest.png)
        ![An example of Random Forest](metricsrandomforest.png)

        We can caputure the results from our run and log the result to Azure Machine Learning. This way we can keep track of the performance of our models while we are experimenting with different models, parameters, data transformations or feature selections. We can specify for ourselfves what is important to track and log number, graphs and tables to Azure ML, including confusion matrices from SKlearn. For a full overview check the [avaiable metrics to track](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments#available-metrics-to-track)

        For a full example on how to run a random forest on Azure Machine Learning and how to log metrics, follow the labs [HERE]

        
    * 15 different sklearn models

        If we set the parameters of `--models` to `sklearnmodels`, we will train 15 different models from the sklearn packages, including randomforeser, naive bayes and Ridge classifier. A full overview of the models being trained, can be founs in the `class Model_choice` on the `sklearn.py` package under `packages`. Comparing different algorithms is possible is different ways. We could submit a new experiment for every algorithm that we try. However, Azure ML offers a better, easier way to manage the exploration of multiple models. This concept is called child runs.  We are going to make use of these child runs. The expiriment will perform a parent run that is going to execute `explore/code/train_15models.py`. Within this file we are going to create child runs. For every of the 15 algoritms that we have we want to create a sub run and log the metrics seprately. Whihin the child run we are going to log the performane and the model .pkl files. This way we can easily track and compare our experiment in Azure ML. If we run this file, the output will look like the following:
        ![An example of tracking accuracy across multiple models](manymodels.png)
        (Note that in this case I am using the new experience, as I believe the new experience is better in tracking child run metrics.)

        For a full example on how to run a multiple model on Azure Machine Learning and how to log metrics and create child runs follow the labs [HERE]

    * Deep Learning

        In this pat we are going to build a Deep Neural Network using Pytorch. If we set the parameters of `--models` to `deeplearning`, we will train this nework. 


*   `train_submit.py`

    This file we use to submit the train script to Azure Machine Learning. This file is created to sumbit different jobs of the training script. At the top of the file, the following paramters we can set to perform training on Azure ML:

    ```python
    # Define comfigs
    # allowed arguments are: randomforest, sklearn, deeplearning
    # randomforest will perform 1 run of randomforest fit
    # sklearnmodels will fit 15 models from sklearn
    # deeplearning will fit a neural network with pytorch
    models = 'sklearnmodels'
    data_local = True
    # if data_local is true, subset is alwats true
    subset = True
    # hyperdrive only works with deeplearning   
    hyperdrive = False
    ```

    Note here the comments. f the set the  `data_local` parameter to `True`, then we need to net the `subset` parameter also to true, because we only have our subset data locally. If you have your full data also available locally, you can of course set this variable to `False`. Nor that we also have a parameter for `hyperdrive`. 

    In this lab, we are going to see how we can levarge Azure ML to do hyeperparameter tuning of our neural network.  Azure Machine Learning can efficiently tune hyperparameters for your model using Azure Machine Learning. Hyperparameter tuning includes the following steps:
    * Define the parameter search space
    * Specify a primary metric to optimize
    * Specify early termination criteria for poorly performing runs
    * Allocate resources for hyperparameter tuning
    * Launch an experiment with the above configuration
    * Visualize the training runs
    * Select the best performing configuration for your model

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

    For a full example on how to run a hyperdrive on Azure Machine Learning and how to log metrics, follow the labs [HERE]

* `score.py`

    The `score.py` file we use when deploying our trained model. The scoring file consist of 3 steps: preprocessing, scoring and postporcoessing.

    The reason that we want to have all these steps together is because they are dependent on each other and make use of the same conda environments. To illustrate an example:

    As a data scientist, I have created a Neural Network with PyTorch. PyTorch is a framework for deep learning. I have created this neural network to perform text classification on emails. A you can image, a neural network cannot read text, but can only read binaries. Therefore, I need to transform my data to binaries. I could let my data engineer do this in an Azure function, but this would not make sense. Because the way my text data is transformed to binaries is dependent on the training data that I have used. This is a highly advanced calculation that follows the same structure as my neural network. Moreover, my neural network is not expecting simple binaries, but binaries of a format called Tensor. The variable format is specific to deep learning frameworks like TensorFlow, Keras and PyTorch. I therefore need the corresponding python libraries. I also need to make sure that the libraries that I use for model training are exactly the same as for preprocessing as most versions of libraries are incompatible with each other (famous example is TesorFlow version 1.* and 2.* and above). And what happens if I have more than 100 different ML models running that all expect the data to arrive in a slightly different manner? We can imagine at this point that this is simply  not feasible to do for a data engineer in an Azure function. A lot of these responsibilities should remain with the data scientist and within the scoring script.

    This is similar to postprocessing. The postprocessing step is highly dependent on the model output. The model will most likely output a binary number or a continuous number. In almost every case, these numbers need to be translated into different outputs like logical names of categories. These transformations are most likely to differ for different models and should therefor also be developed together with the scoring file.
    
    As a last point on this, is that every time that we deploy a new model, we need to do integration test with pre-processing, scoring and post-processing. If we let these steps handle by different tools, these integration test become way harder and the change of failure bigger. So in order to develop and deploy models fast and correct I would highly recommend tot NOT use azure functions for this.
