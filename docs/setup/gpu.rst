.. _running on gpu:

Using GPUs
==========

To run Raster Vision on a realistic dataset in a reasonable amount of time, it is necessary to use a machine with a GPU. Note that Raster Vision will use a GPU if it detects that one is available. If you don't own a machine with a GPU, it is possible to rent one by the minute using a cloud provider such as AWS.

Check that GPU is available
---------------------------

Regardless of how you are running Raster Vision, we recommend you ensure that the GPUs are actually enabled. If you don't, you may run a training job that you think is using the GPU and isn't, and runs very slowly.

One way to check this is to make sure PyTorch can see the GPU(s). To do this, open up a ``python`` console and run the following:

.. code-block:: console

    import torch
    torch.cuda.is_available()
    torch.cuda.get_device_name(0)

This should print out something like:

.. code-block:: console

    True
    Tesla K80

If you have `nvidia-smi <https://developer.nvidia.com/nvidia-system-management-interface>`_  installed, you can also use this command to inspect GPU utilization while the training job is running:

.. code-block:: console

    > watch -d -n 0.5 nvidia-smi

GPUs and Docker
---------------

If you would like to run Raster Vision in a Docker container with GPUs, you'll need to check some things so that the Docker container can utilize the GPUs.

First, you'll need to install the `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ runtime on your system. Follow their `Quickstart <https://github.com/NVIDIA/nvidia-docker#quickstart>`_ and installation instructions. Make sure that your GPU is supported by NVIDIA Docker - if not you might need to find another way to have your Docker container communicate with the GPU. If you figure out how to support more GPUs, please let us know so we can add the steps to this documentation!

When running your Docker container, be sure to include the ``--runtime=nvidia`` option, e.g.

.. code-block:: console

   > docker run --runtime=nvidia --rm -it quay.io/azavea/raster-vision:pytorch-{{ version }} /bin/bash

or use the ``--gpu`` option with the ``docker/run`` script.

.. _aws ec2 setup:

Running on AWS EC2
------------------

The simplest way to run Raster Vision on an AWS GPU is by starting a GPU-enabled EC2 instance such as a p3.2xlarge using the `Deep Learning AMI <https://aws.amazon.com/machine-learning/amis/>`_. We have tested this using the "Deep Learning AMI GPU PyTorch 1.11.0 (Ubuntu 20.04)" with id ``ami-0c968d7ef8a4b0c34``. After SSH'ing into the instance, Raster Vision can be installed with ``pip``, and code can be transferred to this instance with a tool such as ``rsync``.

.. _aws batch setup:

Running on AWS Batch
--------------------

AWS Batch is a service that makes it easier to run Dockerized computation pipelines in the cloud. It starts and stops the appropriate instances automatically and runs jobs sequentially or in parallel according to the dependencies between them. To run Raster Vision using AWS Batch, you'll need to setup your AWS account with a specific set of Batch resources, which you can do using :ref:`cloudformation setup`. After creating the resources on AWS, set the following configuration in your Raster Vision config. Check the AWS Batch console to see the names of the resources that were created, as they vary depending on how CloudFormation was configured.

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

.. seealso::
   For more information about how Raster Vision uses AWS Batch, see the section: :ref:`aws batch`.
