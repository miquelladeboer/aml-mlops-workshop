# aml-mlops-workshop
Azure Machine Learning Workshop using MLOPS

Full end-to-end solution of text classifciation with Pytorch on Azure Machine Learning using MLOps.

## Content 
The resportoty contains te foolowing material:
* [Powerpoint slides with theory](https://github.com/miquelladeboer/aml-mlops-workshop/tree/master/Powerpoints) 
* [Follow along labs to practise](https://github.com/miquelladeboer/aml-mlops-workshop/tree/master/labs)
* [Code with the in between steps of labs](https://github.com/miquelladeboer/aml-mlops-workshop/tree/master/codeperstep)
* [Final code for demo/ example](https://github.com/miquelladeboer/aml-mlops-workshop/tree/master/codefinal)

The labs + final code follow the MLOps guidelines and best practices. For the template, look at: https://github.com/Azure/MLOps-TDSP-Template

In the labs we take 1 dataset and show different capabilities of Azure Machine Learning. We use the 20newgroup dataset from sklearn and we will show the following things:
* Pytorch on Azure Machine Learning
* Run Pytorch Experiment with Horovod distributed training
* Hyperparamter tuning with HyperDrive
* Use GPU's as remote compute to perform distributed training
* Use the concepts of Datasets to work with different versions of the data
* Reuse models from previous run in a new run
* Use Pipelines to combine multiple experiments
* Use model outputs from one step in the pipeline as input in the next step of the pipeline
* Model management
* Model CI/CD
* Model drift monitoring
* Automated Pipeline triggers from model drift


## workshop
In this workshop, you learn about Azure Machine Learning, a cloud-based environment you can use to train, deploy, automate, manage, and track ML models. Azure Machine Learning can be used for any kind of machine learning, from classical ml to deep learning, supervised, and unsupervised learning. Whether you prefer to write Python or R code or zero-code/low-code options such as the designer, you can build, train, and track highly accurate machine learning and deep-learning models in an Azure Machine Learning Workspace.

So what we propose, is a 3-day workshop, covering all the artifact of Azure Machine Learning Service for advanced Data Scientist that want to get to know the Azure Machine Learning Service on a deeply technical level, covering the following subjects:
 
Introduction
* AI Platform
* Azure Machine Learning conceptually
* Azure Machine Learning workspaces
* Azure Machine Learning tools and interfaces
 
Experiment
* Experiment Tracking
* Managing Compute
* Unattended Remote Execution 

Dataset Capabilities
* Event-based Data triggering and ML workflows
  * Data arrives on blob -> Logic App -> Trigger ML Pipeline
* Datastores
* Datasets
* Labeling
 
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
