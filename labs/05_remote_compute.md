## Lab 5: Remote Compute ##
In this lab, we are going to see how we can levarge Azure ML remote compute to submit the HyperDrive run. In this tuturial, we are going to create two different types of compute: CPU and GPU and compare the difference in performance. In this lab we are going to to the following:
* Excecute a run on remote Azure ML CPU cluster
* Excecute a run on remote Azure ML GPU cluster

# Pre-requirements #
1. Completed lab [04_hyperdrive](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/labs/04_hyperdrive.md)
2. Familiarize yourself with the concept of HyperDrive [POWERPOINT]
3. Read the documentation on [How to tune hyperparameters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
4. Read the documentation on [Train Pytorch deep learning models at scale](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch)

## Understand remote compute

A compute target is a designated compute resource/environment where you run your training script or host your service deployment. This location may be your local machine or a cloud-based compute resource. Using compute targets make it easy for you to later change your compute environment without having to change your code.

In a typical model development lifecycle, you might:

* Start by developing and experimenting on a small amount of data. At this stage, we recommend your local environment (local computer or cloud-based VM) as your compute target.

* Scale up to larger data, or do distributed training using one of these training compute targets.

* Once your model is ready, deploy it to a web hosting environment or IoT device with one of these deployment compute targets.
The compute resources you use for your compute targets are attached to a workspace. Compute resources other than the local machine are shared by users of the workspace.

We have done our expemriment fase locally (step 1) and are now moving scale up (step 2). We are going to use Azure Managed compute for this.

Azure Machine Learning Compute is a managed-compute infrastructure that allows the user to easily create a single or multi-node compute. The compute is created within your workspace region as a resource that can be shared with other users in your workspace. The compute scales up automatically when a job is submitted, and can be put in an Azure Virtual Network. The compute executes in a containerized environment and packages your model dependencies in a Docker container. You can use Azure Machine Learning Compute to distribute the training process across a cluster of CPU or GPU compute nodes in the cloud. 

For more info check: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#amlcompute. 

#  Run the code on remote cpu cluster via Azure ML #
To run the `code/modeling/train.py` on remote compute, we need to alter the `train_hyper_submit.py` file, but first we need to create compute.

1. Open the file `infrastructure/create_cpu_compute.py`
    In this file, we are going to create a cpu cluster. In most organizations, data scientist/ml engineers are not allowed to create their own compute. This is mostly done by the IT-deparment due to governacne and compliance reasons. IT-departments can use different, more efficient methods to set up compute clusters. We will discuss this is a later lab when we are talking about Azure ML in the Enterprise. For now, we are going to create the compute ourselves. This can be done easily via the portal or by running the following script. In this script we are going to create a standard `STANDARD_D3_V2` cluster with 4 nodes, called `hypercomputecpu`

2. Run the file `infrastructure/create_cpu_compute.py`
    cpu_cluster_name = "hypercomputecpu"

3. Inspect the compute cluster in the portal

3. In the Estimator in `code/modeling/train_hyper_submit.py`, change the compute target
    Now that we have created the compute cluster, we can change out script to run from `local` compute to the newly created `hypercomputecpu`.

    Take the name for your CPU cluster:

    ```python
    cpu_cluster_name = "hypercomputecpu"
    ```

    Change the `'local'` compute target to:
    `compute_target=workspace.compute_targets[cpu_cluster_name]`

4. Change the experiment name
    So we can keep track of this experiment, we are going to change the experiment name:

    ```python
    # Define the ML experiment
    experiment = Experiment(workspace, "newsgroups_train_hypertune_cpu")
    ```

5. Run the script `train_hyper_submit.py`

6. Go to the portal to inspect the run history

Note: The completed code can be found [here](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/code_labs/modeling/train_hyper_submit_remote_cpu.py)


#  Run the code on remote gpu cluster via Azure ML #
To run the `traindeep.py` on remote compute, we need to alter the `train_hyper_submit.py` file. We are going to create a gpu compute target. In this turutial we choose from the NC-series.

NC-series VMs are powered by the NVIDIA Tesla K80 card and the Intel Xeon E5-2690 v3 (Haswell) processor. Users can crunch through data faster by leveraging CUDA for energy exploration applications, crash simulations, ray traced rendering, deep learning, and more. The NC24r configuration provides a low latency, high-throughput network interface optimized for tightly coupled parallel computing workloads.

We are going to use the `Standard_NC6`

1. Open the file `infrastructure/create_cpu_compute.py`
    In this file, we are going to create a cpu cluster. In most organizations, data scientist/ml engineers are not allowed to create their own compute. This is mostly done by the IT-deparment due to governacne and compliance reasons. IT-departments can use different, more efficient methods to set up compute clusters. We will discuss this is a later lab when we are talking about Azure ML in the Enterprise. For now, we are going to create the compute ourselves. This can be done easily via the portal or by running the following script. In this script we are going to create a standard `STANDARD_D3_V2` cluster with 4 nodes, called `hypercomputecpu`

2. Run the file `infrastructure/create_gpu_compute_hyper.py`
    cpu_cluster_name = "hypercomputegpu"

3. Inspect the compute cluster in the portal

3. In the Estimator in `code/modeling/train_hyper_submit.py`, change the compute target
    Now that we have created the compute cluster, we can change out script to run from `local` compute to the newly created `hypercomputegpu`.

    Take the name for your GPU cluster:

    ```python
    gpu_cluster_name = "hypercomputegpu"
    ```

    Change the `'local'` compute target to:
    `compute_target=workspace.compute_targets[gpu_cluster_name]`

4. Change the experiment name
    So we can keep track of this experiment, we are going to change the experiment name:

    ```python
    # Define the ML experiment
    experiment = Experiment(workspace, "newsgroups_train_hypertune_gpu")
    ```

5. Run the script `train_hyper_submit.py`

6. Go to the portal to inspect the run history

Note: The completed code can be found [here](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/code_labs/modeling/train_hyper_submit_remote_gpu.py)


