.. _running on aws:

Running on AWS
==============

.. _aws ec2 setup:

Running on AWS EC2
------------------

The simplest way to run Raster Vision on an AWS GPU is by starting a GPU-enabled EC2 instance such as a ``p3.2xlarge`` using a `Deep Learning AMI <https://aws.amazon.com/machine-learning/amis/>`_. After SSH'ing into the instance, Raster Vision can be installed with ``pip``, and code can be transferred to this instance with a tool such as ``rsync``.

.. _aws batch setup:

Running on AWS Batch
--------------------

AWS Batch is a service that makes it easier to run Dockerized computation pipelines in the cloud. It starts and stops the appropriate instances automatically and runs jobs sequentially or in parallel according to the dependencies between them. To run Raster Vision using AWS Batch, you'll need to setup your AWS account with a specific set of Batch resources, which you can do using :ref:`cloudformation setup`. After creating the resources on AWS, set the following configuration in your Raster Vision config. Check the AWS Batch console to see the names of the resources that were created, as they vary depending on how CloudFormation was configured.

You can specify configuration options for AWS Batch in multiple ways (see :ref:`raster vision config`).

INI file
~~~~~~~~

.. code:: ini

    [BATCH]
    gpu_job_queue=RasterVisionGpuJobQueue
    gpu_job_def=RasterVisionHostedPyTorchGpuJobDefinition
    cpu_job_queue=RasterVisionCpuJobQueue
    cpu_job_def=RasterVisionHostedPyTorchCpuJobDefinition
    attempts=5

* ``gpu_job_queue`` - job queue for GPU jobs
* ``gpu_job_def`` - job definition that defines the GPU Batch jobs
* ``cpu_job_queue`` - job queue for CPU-only jobs
* ``cpu_job_def`` - job definition that defines the CPU-only Batch jobs
* ``attempts`` - Optional number of attempts to retry failed jobs. It is good to set this to > 1 since Batch often kills jobs for no apparent reason.

Environment variables
~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can set the following environment variables:

.. code:: bash

    GPU_JOB_QUEUE="RasterVisionGpuJobQueue"
    GPU_JOB_DEF="RasterVisionHostedPyTorchGpuJobDefinition"
    CPU_JOB_QUEUE="RasterVisionCpuJobQueue"
    CPU_JOB_DEF="RasterVisionHostedPyTorchCpuJobDefinition"
    ATTEMPTS="5"

* ``GPU_JOB_QUEUE`` - job queue for GPU jobs
* ``GPU_JOB_DEF`` - job definition that defines the GPU Batch jobs
* ``CPU_JOB_QUEUE`` - job queue for CPU-only jobs
* ``CPU_JOB_DEF`` - job definition that defines the CPU-only Batch jobs
* ``ATTEMPTS`` - Optional number of attempts to retry failed jobs. It is good to set this to > 1 since Batch often kills jobs for no apparent reason.

.. seealso::
   For more information about how Raster Vision uses AWS Batch, see the section: :ref:`aws batch`.


.. _aws sagemaker setup:

Running on AWS SageMaker
------------------------

AWS SageMaker is a managed service for training and deploying ML models. For our purposes, it can be thought of as a higher-level alternative to Batch, specialized for ML jobs.

Like Batch, SageMaker allows us to run a DAG of jobs on EC2 instances. But unlike Batch, SageMaker provides a "Pipeline" abstraction which is analogous to :doc:`Raster Vision's own Pipeline <../framework/index>`; running on SageMaker, then, becomes a simple matter of translating the Raster Vision Pipeline to the SageMaker Pipeline. Also unlike Batch, SageMaker does not require setting up job definitions, job queues, and compute environments. Moreover, SageMaker has special support for PyTorch-based training jobs, allowing them to be distributed across multiple nodes. Raster Vision fully supports this functionality which means you can use it to run heavy training loads on multiple machines, each with multiple GPUs.

To use SageMaker, you will need to do the following setup steps:

AWS Service Quotas
~~~~~~~~~~~~~~~~~~

You will need to bump up the service quotas for the kinds of instances you want to use. You can do this from the SageMaker management console like so (as of April 2024):

#. From the drop-down menu under your account name (top-right), select "Service Quotas".
#. From the sidebar, select "AWS Services".
#. Select "Amazon SageMaker".
#. In the search bar, type the instance name e.g. "ml.p3.8xlarge".
#. Select one of the quotas from the search results e.g. "ml.p3.8xlarge for spot training job usage".
#. Click the "Request increase at account level" button, fill out the request form, and submit.

For small increases, these requests are approved automatically.

Local configuration
~~~~~~~~~~~~~~~~~~~

Firstly, you will need to install :mod:`rastervision.aws_sagemaker`:

.. code-block:: console

    > pip install rastervision_aws_sagemaker=={{ version }}


You will also need to specify the following configuration options via an INI file or as environment variables.

INI file
^^^^^^^^

Add the following to your ``~/.rastervision/default`` file.

.. code:: ini

    [SAGEMAKER]
    role=AmazonSageMakerExecutionRole
    cpu_image=123.dkr.ecr.us-east-1.amazonaws.com/raster-vision
    cpu_instance_type=ml.p3.2xlarge
    gpu_image=123.dkr.ecr.us-east-1.amazonaws.com/raster-vision
    gpu_instance_type=ml.p3.2xlarge
    train_image=123.dkr.ecr.us-east-1.amazonaws.com/raster-vision
    train_instance_type=ml.p3.8xlarge
    train_instance_count=2
    use_spot_instances=no
    spot_instance_max_wait_time=86400
    max_run_time=86400

* ``role`` - AWS IAM role with appropriate SageMaker permissions.
* ``cpu_image`` - Docker image URI for CPU jobs.
* ``cpu_instance_type`` - Instance type for CPU jobs.
* ``gpu_image`` - Docker image URI for GPU jobs.
* ``gpu_instance_type`` - Instance type for GPU jobs.
* ``train_image`` - Docker image URI for training jobs. Defaults to ``gpu_image``.
* ``train_instance_type`` - Instance type for training jobs. Defaults to ``gpu_instance_type``.
* ``train_instance_count`` - Number of parallel nodes to run for training jobs. Defaults to 1.
* ``use_spot_instances`` - Whether to use spot instances. Only applies to training jobs.
* ``spot_instance_max_wait_time`` - Maximum time, in seconds, to wait for a spot instance to be allocated. Must be greater than or equal to ``max_run_time``. Default: ``max_run_time``.
* ``max_run_time`` - Maximum job run time in seconds. Default: 86400 (24 hours).


Environment variables
^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can set the following environment variables:

.. code:: bash

    SAGEMAKER_ROLE="AmazonSageMakerExecutionRole"
    SAGEMAKER_CPU_IMAGE="123.dkr.ecr.us-east-1.amazonaws.com/raster-vision"
    SAGEMAKER_CPU_INSTANCE_TYPE="ml.p3.2xlarge"
    SAGEMAKER_GPU_IMAGE="123.dkr.ecr.us-east-1.amazonaws.com/raster-vision"
    SAGEMAKER_GPU_INSTANCE_TYPE="ml.p3.2xlarge"
    SAGEMAKER_TRAIN_IMAGE="123.dkr.ecr.us-east-1.amazonaws.com/raster-vision"
    SAGEMAKER_TRAIN_INSTANCE_TYPE="ml.p3.8xlarge"
    SAGEMAKER_TRAIN_INSTANCE_COUNT="2"
    SAGEMAKER_USE_SPOT_INSTANCES="no"
    SPOT_INSTANCE_MAX_WAIT_TIME="86400"
    MAX_RUN_TIME="86400"

* ``SAGEMAKER_ROLE`` - AWS IAM role with appropriate SageMaker permissions.
* ``SAGEMAKER_CPU_IMAGE`` - Docker image URI for CPU jobs.
* ``SAGEMAKER_CPU_INSTANCE_TYPE`` - Instance type for CPU jobs.
* ``SAGEMAKER_GPU_IMAGE`` - Docker image URI for GPU jobs.
* ``SAGEMAKER_GPU_INSTANCE_TYPE`` - Instance type for GPU jobs.
* ``SAGEMAKER_TRAIN_IMAGE`` - Docker image URI for training jobs. Defaults to ``SAGEMAKER_GPU_IMAGE``.
* ``SAGEMAKER_TRAIN_INSTANCE_TYPE`` - Instance type for training jobs. Defaults to ``SAGEMAKER_GPU_INSTANCE_TYPE``.
* ``SAGEMAKER_TRAIN_INSTANCE_COUNT`` - Number of parallel nodes to run for training jobs. Defaults to 1.
* ``SAGEMAKER_USE_SPOT_INSTANCES`` - Whether to use spot instances. Only applies to training jobs.
* ``SPOT_INSTANCE_MAX_WAIT_TIME`` - Maximum time, in seconds, to wait for a spot instance to be allocated. Must be greater than or equal to ``MAX_RUN_TIME``. Default: ``MAX_RUN_TIME``.
* ``MAX_RUN_TIME`` - Maximum job run time in seconds. Default: 86400 (24 hours).


.. seealso::
   For more information about how Raster Vision uses AWS SageMaker, see the section: :ref:`aws sagemaker`.
