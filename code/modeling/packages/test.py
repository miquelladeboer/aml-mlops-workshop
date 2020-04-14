# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Definition for a generic Estimator class which support training with any learning framework.

Estimators are the building blocks for training. An estimator encapsulates training code and parameters,
compute resources, and runtime environment for a particular training scenario. With Azure Machine Learning,
you can use the :class:`azureml.core.runconfig.RunConfiguration` and
:class:`azureml.core.script_run_config.ScriptRunConfig`objects to easily submit training scripts to compute
targets. Using the configuration pattern gives you flexibility and control when training.

The Azure Machine Learning Python SDK facilitates deep learning model training with an alternative
higher-level abstraction, the Estimator. The Estimator class allows you to construct run configurations
and to submit training scripts using any learning framework you choose. You can submit your run on any
compute target, whether it's a local machine, a single VM in Azure, or a GPU cluster in Azure. Azure
Machine Learning also provides :class:`azureml.train.dnn.PyTorch`, :class:`azureml.train.dnn.TensorFlow`,
:class:`azureml.train.dnn.Chainer`, and :class:`azureml.train.sklearn.SKLearn` estimators to simplify using
these frameworks.

For an introduction to model training, see `How to train models with Azure using an estimator
<https://docs.microsoft.com/azure/machine-learning/how-to-train-ml-models>`_.  For
more information about Docker containers used in Azure ML training, see https://github.com/Azure/AzureML-Containers.

"""

import logging

from azureml.core._experiment_method import experiment_method
from azureml.train._estimator_helper import _estimator_submit_method, _init_run_config, \
    _is_notebook_run, _update_config_for_notebook_run
from packages.aml_helpers import MMLBase

module_logger = logging.getLogger(__name__)


class Estimator1(MMLBase):
    """Represents a generic estimator to train data using any supplied framework.

    This class is designed for use with machine learning frameworks that do not already have an Azure Machine Learning
    pre-configured estimator. Pre-configured estimators exist for :class:`azureml.train.dnn.Chainer`,
    :class:`azureml.train.dnn.PyTorch`, :class:`azureml.train.dnn.TensorFlow`, and
    :class:`azureml.train.sklearn.SKLearn`.

    The Estimator class wraps run configuration information to help simplify the tasks of specifying how a
    script is executed. It supports single-node as well as multi-node execution. Running the estimator
    produces a model in the output directory specified in your training script.

    .. remarks::
            The example below shows how to create an estimator for training using the
            `Microsoft Cognitive Toolkit (CNTK) <https://docs.microsoft.com/cognitive-toolkit/>`_. CNTK
            doesn't have a corresponding custom estimator defined in Azure ML SDK but can still be used for
            training with the Estimator class.

            .. code-block:: python

                from azureml.train.estimator import Estimator

                script_params = {
                    '--num_epochs': 20,
                    '--data_dir': ds_data.as_mount(),
                    '--output_dir': './outputs'
                }

                estimator = Estimator(source_directory=project_folder,
                                      compute_target=compute_target,
                                      entry_script='cntk_distr_mnist.py',
                                      script_params=script_params,
                                      node_count=2,
                                      process_count_per_node=1,
                                      distributed_backend='mpi',
                                      pip_packages=['cntk-gpu==2.6'],
                                      custom_docker_image='microsoft/mmlspark:gpu-0.12',
                                      use_gpu=True)

            Full sample is available from
            https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/training-with-deep-learning/distributed-cntk-with-custom-docker/distributed-cntk-with-custom-docker.ipynb


            For more information about training with Estimator, see the tutorial
            `Train models with Azure Machine Learning using estimator
            <https://docs.microsoft.com/azure/machine-learning/how-to-train-ml-models>`_.

            For information about Docker containers used in Azure ML training, see
            https://github.com/Azure/AzureML-Containers.

    :param source_directory: A local directory containing experiment configuration and code files needed for a
        training job.
    :type source_directory: str
    :param compute_target:  The compute target where training will happen. This can either be an object or the
        string "local".
    :type compute_target: azureml.core.compute_target.AbstractComputeTarget or str
    :param vm_size: The VM size of the compute target that will be created for the training. Supported values:
        Any `Azure VM size
        <https://docs.microsoft.com/azure/cloud-services/cloud-services-sizes-specs>`_.
    :type vm_size: str
    :param vm_priority: The VM priority of the compute target that will be created for the training. If not
        specified, 'dedicated' is used.

        Supported values: 'dedicated' and 'lowpriority'.

        This takes effect only when the ``vm_size`` parameter is specified in the input.
    :type vm_priority: str
    :param entry_script: The relative path to the file used to start training.
    :type entry_script: str
    :param script_params: A dictionary of command-line arguments to pass to the training script specified in
        ``entry_script``.
    :type script_params: dict
    :param node_count: The number of nodes in the compute target used for training. If greater than 1, an MPI
        distributed job will be run.
    :type node_count: int
    :param process_count_per_node: The number of processes (or "workers") to run on each node. If greater than 1,
        an MPI distributed job will be run. Only the :class:`azureml.core.compute.AmlCompute` target is supported for
        distributed jobs.
    :type process_count_per_node: int
    :param distributed_backend: The communication backend for distributed training.

        DEPRECATED. Use the ``distributed_training`` parameter.

        Supported values: 'mpi'. 'mpi' represents MPI/Horovod.

        This parameter is required when ``node_count`` or ``process_count_per_node`` > 1.

        When ``node_count`` == 1 and ``process_count_per_node`` == 1, no backend will be used
        unless the backend is explicitly set. Only the :class:`azureml.core.compute.AmlCompute` target is supported for
        distributed training.
    :type distributed_backend: str
    :param distributed_training: Parameters for running a distributed training job.

        For running a distributed job with MPI backend, use :class:`azureml.train.estimator.Mpi`
        object to specify ``process_count_per_node``.
    :type distributed_training: azureml.train.estimator.Mpi
    :param use_gpu: Indicates whether the environment to run the experiment should support GPUs.
        If true, a GPU-based default Docker image will be used in the environment. If false, a CPU-based
        image will be used. Default Docker images (CPU or GPU) will be used only if the ``custom_docker_image``
        parameter is not set. This setting is used only in Docker enabled compute targets.
    :type use_gpu: bool
    :param use_docker: Specifies whether the environment to run the experiment should be Docker-based.
    :type use_docker: bool
    :param custom_docker_base_image: The name of the Docker image from which the image to use for training
        will be built.

        DEPRECATED. Use the ``custom_docker_image`` parameter.

        If not set, a default CPU-based image will be used as the base image.
    :type custom_docker_base_image: str
    :param custom_docker_image: The name of the Docker image from which the image to use for training
        will be built. If not set, a default CPU-based image will be used as the base image. Only specify
        images available in public docker repositories (Docker Hub). To use an image from a private docker
        repository, use the constructor's ``environment_definition`` parameter instead.
    :type custom_docker_image: str
    :param image_registry_details: The details of the Docker image registry.
    :type image_registry_details: azureml.core.container_registry.ContainerRegistry
    :param user_managed: Specifies whether Azure ML reuses an existing Python environment. If false,
        a Python environment is created based on the conda dependencies specification.
    :type user_managed: bool
    :param conda_packages: A list of strings representing conda packages to be added to the Python environment
        for the experiment.
    :type conda_packages: list
    :param pip_packages: A list of strings representing pip packages to be added to the Python environment
        for the experiment.
    :type pip_packages: list
    :param conda_dependencies_file_path: The relative path to the conda dependencies yaml file.
        If specified, Azure ML will not install any framework related packages.

        DEPRECATED. Use the ``conda_dependencies_file`` paramenter.

        Specify either ``conda_dependencies_file_path`` or ``conda_dependencies_file``. If both are specified,
        ``conda_dependencies_file`` is used.
    :type conda_dependencies_file_path: str
    :param pip_requirements_file_path: The relative path to the pip requirements text file.

        DEPRECATED. Use the ``pip_requirements_file`` parameter.

        This parameter can be specified in combination with the ``pip_packages`` parameter. Specify either
        ``pip_requirements_file_path`` or ``pip_requirements_file``. If both are specified,
        ``pip_requirements_file`` is used.
    :type pip_requirements_file_path: str
    :param conda_dependencies_file: The relative path to the conda dependencies yaml file.
        If specified, Azure ML will not install any framework related packages.
    :type conda_dependencies_file: str
    :param pip_requirements_file: The relative path to the pip requirements text file.
        This parameter can be specified in combination with the ``pip_packages`` parameter.
    :type pip_requirements_file: str
    :param environment_variables: A dictionary of environment variables names and values.
        These environment variables are set on the process where user script is being executed.
    :type environment_variables: dict
    :param environment_definition: The environment definition for the experiment. It includes
        PythonSection, DockerSection, and environment variables. Any environment option not directly
        exposed through other parameters to the Estimator construction can be set using this
        parameter. If this parameter is specified, it will take precedence over other environment-related
        parameters like ``use_gpu``, ``custom_docker_image``, ``conda_packages``, or ``pip_packages``.
        Errors will be reported on invalid combinations.
    :type environment_definition: azureml.core.Environment
    :param inputs: A list of :class:`azureml.data.data_reference.DataReference` or
        :class:`azureml.data.dataset_consumption_config.DatasetConsumptionConfig` objects to use as input.
    :type inputs: list
    :param source_directory_data_store: The backing data store for the project share.
    :type source_directory_data_store: azureml.core.Datastore
    :param shm_size: The size of the Docker container's shared memory block. If not set, the default
        azureml.core.environment._DEFAULT_SHM_SIZE is used. For more information, see
        `Docker run reference <https://docs.docker.com/engine/reference/run/>`_.
    :type shm_size: str
    :param resume_from: The data path containing the checkpoint or model files from which to resume the experiment.
    :type resume_from: azureml.data.datapath.DataPath
    :param max_run_duration_seconds: The maximum allowed time for the run. Azure ML will attempt to automatically
        cancel the run if it take longer than this value.
    :type max_run_duration_seconds: int
    """

    _SUPPORTED_BACKENDS = ["mpi"]

    @experiment_method(submit_function=_estimator_submit_method)
    def __init__(self,
                 source_directory,
                 *,
                 compute_target=None,
                 vm_size=None,
                 vm_priority=None,
                 entry_script=None,
                 script_params=None,
                 node_count=1,
                 process_count_per_node=1,
                 distributed_backend=None,
                 distributed_training=None,
                 use_gpu=False,
                 use_docker=True,
                 custom_docker_base_image=None,
                 custom_docker_image=None,
                 image_registry_details=None,
                 user_managed=False,
                 conda_packages=None,
                 pip_packages=None,
                 conda_dependencies_file_path=None,
                 pip_requirements_file_path=None,
                 conda_dependencies_file=None,
                 pip_requirements_file=None,
                 environment_variables=None,
                 environment_definition=None,
                 inputs=None,
                 source_directory_data_store=None,
                 shm_size=None,
                 resume_from=None,
                 max_run_duration_seconds=None,
                 _disable_validation=False,
                 _show_lint_warnings=True,
                 _show_package_warnings=False):
        """Initialize the estimator.

        :param source_directory: A local directory containing experiment configuration and code files needed for a
           training job.
        :type source_directory: str
        :param compute_target: The compute target where training will happen. This can either be an object or the
            string "local".
        :type compute_target: azureml.core.compute_target.AbstractComputeTarget or str
        :param vm_size: The VM size of the compute target that will be created for the training. Supported values:
            Any `Azure VM size
            <https://docs.microsoft.com/azure/cloud-services/cloud-services-sizes-specs>`_.
        :type vm_size: str
        :param vm_priority: The VM priority of the compute target that will be created for the training. If not
            specified, 'dedicated' is used.

            Supported values: 'dedicated' and 'lowpriority'.

            This takes effect only when the ``vm_size`` parameter is specified in the input.
        :type vm_priority: str
        :param entry_script: The relative path to the file used to start training.
        :type entry_script: str
        :param script_params: A dictionary of command-line arguments to pass to the training script specified in
           ``entry_script``.
        :type script_params: dict
        :param node_count: The number of nodes in the compute target used for training. If greater than 1, a MPI
            distributed job will be run. Only the :class:`azureml.core.compute.AmlCompute` target is supported for
            distributed jobs.
        :type node_count: int
        :param process_count_per_node: The number of processes per node. If greater than 1, a MPI
             distributed job will be run. Only the :class:`azureml.core.compute.AmlCompute` target is supported
             for distributed jobs.
        :type process_count_per_node: int
        :param distributed_backend: The communication backend for distributed training.

            DEPRECATED. Use the ``distributed_training`` parameter.

            Supported values: 'mpi'. 'mpi' represents MPI/Horovod.

            This parameter is required when ``node_count`` or ``process_count_per_node`` > 1.

            When ``node_count`` == 1 and ``process_count_per_node`` == 1, no backend will be used
            unless the backend is explicitly set. Only the :class:`azureml.core.compute.AmlCompute` target is
            supported for distributed training.
        :type distributed_backend: str
        :param distributed_training: Parameters for running a distributed training job.

            For running a distributed job with MPI backend, use :class:`azureml.train.estimator.Mpi`
            object to specify ``process_count_per_node``.
        :type distributed_training: azureml.train.estimator.Mpi
        :param use_gpu: Specifies whether the environment to run the experiment should support GPUs.
            If true, a GPU-based default Docker image will be used in the environment. If false, a CPU-based
            image will be used. Default Docker images (CPU or GPU) will be used only if the ``custom_docker_image``
            parameter is not set. This setting is used only in Docker-enabled compute targets.
        :type use_gpu: bool
        :param use_docker: Specifies whether the environment to run the experiment should be Docker-based.
        :type use_docker: bool
        :param custom_docker_base_image: The name of the Docker image from which the image to use for training
            will be built.

            DEPRECATED. Use the ``custom_docker_image`` parameter.

            If not set, a default CPU-based image will be used as the base image.
        :type custom_docker_base_image: str
        :param custom_docker_image: The name of the Docker image from which the image to use for training
            will be built. If not set, a default CPU-based image will be used as the base image. Only specify
            images available in public docker repositories (Docker Hub). To use an image from a private docker
            repository, use the constructor's ``environment_definition`` parameter instead.
        :type custom_docker_image: str
        :param image_registry_details: The details of the Docker image registry.
        :type image_registry_details: azureml.core.container_registry.ContainerRegistry
        :param user_managed: Specifies whether Azure ML reuses an existing Python environment. If false,
            a Python environment is created based on the conda dependencies specification.
        :type user_managed: bool
        :param conda_packages: A list of strings representing conda packages to be added to the Python environment
            for the experiment.
        :type conda_packages: list
        :param pip_packages: A list of strings representing pip packages to be added to the Python environment
            for the experiment.
        :type pip_packages: list
        :param conda_dependencies_file_path: The relative path to the conda dependencies yaml file.
            If specified, Azure ML will not install any framework related packages.

            DEPRECATED. Use the ``conda_dependencies_file`` paramenter.

            Specify either ``conda_dependencies_file_path`` or ``conda_dependencies_file``. If both are specified,
            ``conda_dependencies_file`` is used.
        :param pip_requirements_file_path: The relative path to the pip requirements text file.

            DEPRECATED. Use the ``pip_requirements_file`` parameter.

            This can be provided in combination with the ``pip_packages`` parameter. Specify either
            ``pip_requirements_file_path`` or ``pip_requirements_file``. If both are specified,
            ``pip_requirements_file`` is used.
        :type pip_requirements_file_path: str
         :param conda_dependencies_file: The relative path to the conda dependencies yaml file.
            If specified, Azure ML will not install any framework related packages.
        :type conda_dependencies_file: str
        :param pip_requirements_file: The relative path to the pip requirements text file.
            This can be provided in combination with the ``pip_packages`` parameter.
        :type pip_requirements_file: str
        :param environment_variables: A dictionary of environment variables names and values.
            These environment variables are set on the process where user script is being executed.
        :type environment_variables: dict
        :param environment_definition: The environment definition for the experiment. It includes
            PythonSection, DockerSection, and environment variables. Any environment option not directly
            exposed through other parameters to the Estimator construction can be set using this
            parameter. If this parameter is specified, it will take precedence over other environment-related
            parameters like ``use_gpu``, ``custom_docker_image``, ``conda_packages``, or ``pip_packages``.
            Errors will be reported on invalid combinations.
        :type environment_definition: azureml.core.Environment
        :param inputs: A list of :class:`azureml.data.data_reference.DataReference` or
            :class:`azureml.data.dataset_consumption_config.DatasetConsumptionConfig` objects to use as input.
        :type inputs: list
        :param source_directory_data_store: The backing data store for the project share.
        :type source_directory_data_store: azureml.core.Datastore
        :param shm_size: The size of the Docker container's shared memory block. If not set, the default
        azureml.core.environment._DEFAULT_SHM_SIZE is used. For more information, see
        `Docker run reference <https://docs.docker.com/engine/reference/run/>`_.
        :type shm_size: str
        :param resume_from: The data path containing the checkpoint or model files from which to resume the experiment.
        :type resume_from: azureml.data.datapath.DataPath
        :param max_run_duration_seconds: The maximum allowed time for the run. Azure ML will attempt to automatically
            cancel the run if it takes longer than this value.
        :type max_run_duration_seconds: int
        :param _disable_validation: Disable script validation before run submission.
        :type _disable_validation: bool
        :param _show_lint_warnings: Show script linting warnings.
        :type _show_lint_warnings: bool
        :param _show_package_warnings: Show package validation warnings.
        :type _show_package_warnings: bool
        """
        estimator_config = _init_run_config(
            estimator=self,
            source_directory=source_directory,
            compute_target=compute_target,
            vm_size=vm_size,
            vm_priority=vm_priority,
            entry_script=entry_script,
            script_params=script_params,
            node_count=node_count,
            process_count_per_node=process_count_per_node,
            distributed_backend=distributed_backend,
            distributed_training=distributed_training,
            use_gpu=use_gpu,
            use_docker=use_docker,
            custom_docker_base_image=custom_docker_base_image,
            custom_docker_image=custom_docker_image,
            image_registry_details=image_registry_details,
            user_managed=user_managed,
            conda_packages=conda_packages,
            pip_packages=pip_packages,
            conda_dependencies_file_path=conda_dependencies_file_path,
            pip_requirements_file_path=pip_requirements_file_path,
            conda_dependencies_file=conda_dependencies_file,
            pip_requirements_file=pip_requirements_file,
            environment_variables=environment_variables,
            environment_definition=environment_definition,
            inputs=inputs,
            source_directory_data_store=source_directory_data_store,
            shm_size=shm_size,
            resume_from=resume_from,
            max_run_duration_seconds=max_run_duration_seconds)

        self._manual_restart_used = (resume_from is not None)
        self._distributed_backend = distributed_backend
        self._disable_validation = _disable_validation
        self._show_lint_warnings = _show_lint_warnings
        self._show_package_warnings = _show_package_warnings
        if distributed_training:
            self._distributed_backend = distributed_training

        if _is_notebook_run(estimator_config.script):
            _update_config_for_notebook_run(estimator_config,
                                            use_gpu,
                                            custom_docker_image)

        super(self.__class__, self).__init__(source_directory,
                                             compute_target=compute_target,
                                             estimator_config=estimator_config)
