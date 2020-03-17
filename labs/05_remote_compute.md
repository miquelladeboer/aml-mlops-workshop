## Lab 5: Remote Compute ##
In this lab, we are going to see how we can levarge Azure ML remote compute to submit the HyperDrive run. In this tuturial, we are going to create two different types of compute: CPU and GPU and compare the difference in performance.

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

#  Run the code via Azure ML #
To run the `traindeep.py` on remote compute, we need to alter the `traindeep_submit.py` file.

1. Add the required libraries from AzureML
    ```
    from azureml.core.compute import ComputeTarget, AmlCompute
    from azureml.core.compute_target import ComputeTargetException
    ```
2. Create a compute target
    First, we are going to create a cpu compute target. In this turutial we choose the `'STANDARD_D3_V2'`, but you can choose the compute target you need for your solution. To see all supported VM sizes, run: `print(AmlCompute.supported_vmsizes(workspace=ws))`

    ```
    # Create compute target if not present
    # Choose a name for your CPU cluster
    cpu_cluster_name = "hypercomputecpu"

    # Verify that cluster does not exist already
    try:
        cpu_cluster = ComputeTarget(workspace=workspace, name=cpu_cluster_name)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D3_V2',
                                                           max_nodes=4)
        cpu_cluster = ComputeTarget.create(workspace, cpu_cluster_name,
                                       compute_config)

    cpu_cluster.wait_for_completion(show_output=True)
    ```

3. In the Estimator, change the compute target
    Change the `'local'` compute target to:
    `compute_target=workspace.compute_targets[cpu_cluster_name]`

4. Run the script `traindeep_submit.py`

5. Go to the portal to inspect the run history

Note: the correct code is already available in codeazureml in the script `explore\traindeep_submit_remote.py`. In here, all ready to use code is available for the entire workshop.

