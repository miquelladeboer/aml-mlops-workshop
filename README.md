# aml-mlops-workshop
Azure Machine Learning Workshop using MLOPS (WIP)

Full end-to-end solution of text classifciation with Pytorch on Azure Machine Learning using MLOps covering:

* Distibuted Data Engineering usgin Dask on Azure Machine Learning
* Distributed Hyperparameter tuning using Pytorch on Azure Machine Learning
* Data Science Code Testing
* Datastores and Datasets for data governance and management
* MLOps ARM templates for configuring environment
* YAML for Run Configurations and ML Pipelines
* CI/CD pipelines for model deployment
* Monitoring models using Application Insights
* Feedback Loop with Active Learning based on Model Drift

# Content 
The resportoty contains te foolowing material:
* [Powerpoint slides with theory](https://github.com/miquelladeboer/aml-mlops-workshop/tree/master/Powerpoints) 
* [Follow along labs to practise](https://github.com/miquelladeboer/aml-mlops-workshop/tree/master/labs)
* [Final code for demo/ example](https://github.com/miquelladeboer/aml-mlops-workshop/tree/master/code)
* [Quickstart template](https://github.com/miquelladeboer/aml-mlops-workshop/tree/master/template)

The labs + final code follow the MLOps guidelines and best practices. For the template, look at: https://github.com/Azure/MLOps-TDSP-Template

In the lab we take 1 use-case and show different capabilities of Azure Machine Learning. 

# Quick start

To quickstart the solution, follow the [quickstart template](https://github.com/miquelladeboer/aml-mlops-workshop/tree/master/quickstart.md)




# workshop
In this workshop, you learn about Azure Machine Learning, a cloud-based environment you can use to train, deploy, automate, manage, and track ML models. Azure Machine Learning can be used for any kind of machine learning, from classical ml to deep learning, supervised, and unsupervised learning. Whether you prefer to write Python or R code or zero-code/low-code options such as the designer, you can build, train, and track highly accurate machine learning and deep-learning models in an Azure Machine Learning Workspace.

So what we propose, is a 3-day workshop, covering all the artifact of Azure Machine Learning Service for advanced Data Scientist that want to get to know the Azure Machine Learning Service on a deeply technical level, covering the following subjects:
 
## Introduction
* AI Platform
* Azure Machine Learning conceptually
* Azure Machine Learning workspaces
* Azure Machine Learning tools and interfaces

[PPT Introduction to Azure Machine Learning](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/Powerpoints/Module%201%20-%20Introduction.pptx) 

[Lab 1: Setup](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/labs/01_setup.md)
 
## Experiment
* Experiment Tracking
* Unattended Remote Execution (parallel with child runs)

[PPT Experiments on Azure Machine Learning](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/Powerpoints/Module%202%20-%20Experiments.pptx) 

[Lab 2: Experiment Tracking](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/labs/02_experiment.md)

[Lab 3: Unattended Remote Execution (parallel)](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/labs/03_childrun.md)

## Hyperparameter Tuning
* Deep Learning with Pytorch
* Hyperparameter tuning

[PPT Hyperparameter Tuning on Azure Machine Learning](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/Powerpoints/Module%203%20-%20Hyperparameter%20tuning.pptx) 

[Lab 4: Hyperparameter Tuning](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/labs/04_hyperdrive)

## Remote Compute

[PPT Hyperparameter Tuning on Azure Machine Learning](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/Powerpoints/Module%203%20-%20Hyperparameter%20tuning.pptx) 

[Lab 4: Hyperparameter Tuning](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/labs/04_hyperdrive)


## Dataset Capabilities
* Datastores
* Datasets

 
Pipelines
* Pipelines conceptually
* Pipeline Experiments
* Publishing pipelines
 
Model Management
* Artifact Store
* Model CI/CD
* Monitoring
 
Azure ML in the Enterprise
* Governance
* RBAC
* Custom Roles

# Get started
To get started, follow the setup file [01_setup](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/labs/01_setup.md)
