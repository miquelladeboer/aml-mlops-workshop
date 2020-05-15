# Lab 0: setting up the environment

In this first lab, we'll set up our working environment.

## Requirements

* Visual Studio Code
  Download and Install [Visual Studio Code](https://code.visualstudio.com/)

* Miniconda
  Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

* Create a new virtual environment
From a command line window, run  `conda create -n myenvworkshop python=3.6 anaconda` 

* Activate the virtual environment
From a command line window, run  `conda activate myenvworkshop` 

* Azure ML SDK
  From a command line window, run the following command to install the python client package for Azure ML: `pip install azureml-sdk`

* Azure ML SDK for Notebooks
  From a command line window, run the following command to install the python client package for Azure ML: `pip install azureml-sdk[notebooks]`

* Install frameworks for deeplearning: Pytorch
  From a command line window, run `conda install -c pytorch pytorch`

* Install frameworks for distributed data preperation: Dask
  From a command line window, run `pip install --upgrade dask distributed`

* Azure CLI
  From a command line window, run the following command to install the Azure CLI, used for authentication and management tasks: `pip install azure-cli`


* A git client to clone the lab content
  For example Git SCM - https://git-scm.com/.

## Clone the repository

Clone the following git repository: git clone  https://github.com/miquelladeboer/workshop-azure-machine-learning

## Open the cloned git repository in VS Code or your favorite IDE

## Install the required packahes

* Install needed packages
  From a command line window, run `pip install -r requirements.txt`

## Az Login
From a terminal, login to your subscription on Azure using the azure cli.

* `az login`

If you have multiple subscriptions, you might want to set the right subscription by using the following command. 

* `az account set -s <subid>`

## Deploy an ML workspace and dependent resources 

In the script `infrastructure/create_mlworkspace.py`, change the subscription id to you Azure subscription. 
Execute the script `infrastructure/create_mlworkspace.py` to deploy the ML workspace resource and dependent resources such as a Keyvault instance and a Storage Account.

## Browse through the created resources in the portal

You can now take a look over the created resources via the [Azure Portal](http://portal.azure.com/).
