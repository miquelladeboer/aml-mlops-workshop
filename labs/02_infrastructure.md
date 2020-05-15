## lab 2: Infrastructure

In this tutorial ou will learn about how you could use [Azure Pipelines](https://azure.microsoft.com/en-us/services/devops/pipelines/) for the automated deployment of infrastructure. This way of working enables you to incrementally deploy changes to your resources, stage the changes over different environments, and build confidence as your system growths more complex.

# Pre-requirements #
1. If not familiair with Infrastructure as Code read the  [IaC documentation](https://docs.microsoft.com/en-us/azure/devops/learn/what-is-infrastructure-as-code)
2. Finished the introduction file [01_introduction](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/labs/01_introduction.md)

### Getting started

Complete the below steps to set up your pipeline for infrastructure roll out:

* Navigate to [Azure DevOps](http://dev.azure.com/) and create a new organization and project. You can also re-use an existing organization and/or project.
* Create a new [service connection](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/service-endpoints?view=azure-devops&tabs=yaml) in Azure DevOps of the Azure Resources Manager connection type. Azure DevOps will authenticate using this connection to make deployments to your Azure Subscription.
* In [deploy-infra.yml](build-and-release/deploy-infra.yml) replace `<your-service-connection-name>` by the name of the service connection that you created in the previous step.
* Some Azure resources require you to use globally unique names across Azure. This holds for example for storage account resources. Adapt resource names in the ARM parameter files to a name that is globally unique. Note that you should update the parameter files for the ML workspace and ML compute resources as well once you update the names of the underlying resources.
* Make a test deployment using the provided powershell script `deploy-infra.ps1`.
* Set up a new pipeline in Azure DevOps with the option to re-use an existing template. Point to the pipeline definition [deploy-infra.yml](build-and-release/deploy-infra.yml) in your repository.
* Run your pipeline from Azure DevOps. On completion, you should see a result like the below.
![An example of a pipeline for Infrastructure roll out](infra_stages.png)
