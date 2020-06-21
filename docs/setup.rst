.. _rv2_setup:

Setup
=====

.. _rv2_docker images:

Docker Images
-----------------

Using the Docker images published for Raster Vision makes it easy to use a fully set up environment. We have tested this with Docker 18, although you may be able to use a lower version.

The images we publish include plugins and dependencies for using Raster Vision with PyTorch, and AWS S3 and Batch. These are published to `quay.io/azavea/raster-vision <https://quay.io/repository/azavea/raster-vision>`_.  To run the container for the latest release, run:

.. code-block:: console

   > docker run --rm -it quay.io/azavea/raster-vision:pytorch-0.12 /bin/bash

There are also images with the `-latest` suffix for the latest commits on the ``master`` branch. You'll likely need to mount volumes and expose ports to make this container fully useful; see the `docker/run <https://github.com/azavea/raster-vision/blob/0.12/docker/run>`_ script for an example usage.

You can also base your own Dockerfiles off the Raster Vision image to use with your own codebase. See the Dockerfiles in the `Raster Vision Examples <https://github.com/azavea/raster-vision-examples>`_ repository for an example.

Docker Scripts
~~~~~~~~~~~~~~

There are several scripts under `docker/ <https://github.com/azavea/raster-vision/tree/0.12/docker>`_ in the Raster Vision repo that make it easier to build the Docker images from scratch, and run the container in various ways. These are useful if you are experimenting with changes to the Raster Vision source code.

After cloning the repo, you can build the Docker image using:

.. code-block:: console

    > docker/build --pytorch

Before running the container, set an environment variable to a local directory in which to store data.

.. code-block:: console

    > export RASTER_VISION_DATA_DIR="/path/to/data"

To run a Bash console in the PyTorch Docker container use:

.. code-block:: console

    > docker/run

This will mount the ``$RASTER_VISION_DATA_DIR`` local directory to to ``/opt/data/`` inside the container.

This script also has options for forwarding AWS credentials, running Jupyter notebooks, and switching between different images, which can be seen below.

.. code-block:: console

    > ./docker/run --help
    Usage: run <options> <command>

    Run a console in a Raster Vision Docker image locally.
    By default, the raster-vision-pytorch image is used in the CPU runtime.

    Environment variables:
    RASTER_VISION_DATA_DIR (directory for storing data; mounted to /opt/data)
    AWS_PROFILE (optional AWS profile)
    RASTER_VISION_REPO (optional path to main RV repo; mounted to /opt/src)

    Options:
    --aws forwards AWS credentials (sets AWS_PROFILE env var and mounts ~/.aws to /root/.aws)
    --tensorboard maps port 6006
    --gpu use the NVIDIA runtime and GPU image
    --name sets the name of the running container
    --jupyter forwards port 8888, mounts ./notebooks to /opt/notebooks, and runs Jupyter
    --debug maps port 3007 on localhost to 3000 inside container

    All arguments after above options are passed to 'docker run'.

.. _rv2_install raster vision:

Installing via pip
------------------------

.. currentmodule:: rastervision

Rather than running Raster Vision from inside a Docker container, you can directly install the library using ``pip``. However, we recommend using the Docker images since it can be difficult to install some of the dependencies.

.. code-block:: console

   > pip install rastervision==0.12.0

.. note:: Raster Vision requires Python 3 or later. Use ``pip3 install rastervision==0.12.0`` if you have more than one version of Python installed.

You will also need various dependencies that are not pip-installable. For an example of setting these up, see the various `Dockerfiles  <https://github.com/azavea/raster-vision/blob/0.12/>`_.

Troubleshooting macOS Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter problems running ``pip install rastervision==0.12.0`` on macOS, you may have to manually install Cython and pyproj.

To circumvent a problem installing pyproj with Python 3.7, you may also have to install that library using ``git+https``:

.. code-block:: console

  > pip install cython
  > pip install git+https://github.com/jswhit/pyproj.git@e56e879438f0a1688b89b33228ebda0f0d885c19
  > pip install rastervision==0.12.0

.. _rv2_raster vision config:

Raster Vision Configuration
---------------------------

Raster Vision is configured via the `everett <https://everett.readthedocs.io/en/latest/index.html>`_ library, and will look for configuration in the following locations, in this order:

* Environment Variables
* A ``.env`` file in the working directory that holds environment variables.
* Raster Vision INI configuration files

By default, Raster Vision looks for a configuration file named ``default`` in the ``${HOME}/.rastervision`` folder.

Profiles
~~~~~~~~

Profiles allow you to specify profile names from the command line or environment variables
to determine which settings to use. The configuration file used will be named the same as the
profile: if you had two profiles (the ``default`` and one named ``myprofile``), your
``${HOME}/.rastervision`` would look like this:

.. code-block:: console

   > ls ~/.rastervision
   default    myprofile

Use the ``rastervision --profile`` option in the :ref:`rv2_cli` to set the profile.

Configuration File Sections
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _rv2_s3 config:

AWS_S3
^^^^^^

.. code-block:: ini

   [AWS_S3]
   requester_pays = False

* ``requester_pays`` - Set to True if you would like to allow using `requester pays <https://docs.aws.amazon.com/AmazonS3/latest/dev/RequesterPaysBuckets.html>`_ S3 buckets. The default value is False.

Other
^^^^^^

Other configurations are documented elsewhere:

* :ref:`rv2_aws batch config`

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Any profile file option can also be stated in the environment. Just prepend the section name to the setting name, e.g. ``export AWS_S3_REQUESTER_PAYS="False"``.

In addition to those environment variables that match the INI file values, there are the following environment variable options:

* ``TMPDIR`` - Setting this environment variable will cause all temporary directories to be created inside this folder. This is useful, for example, when you have a Docker container setup that mounts large network storage into a specific directory inside the Docker container. The tmp_dir can also be set on :ref:`rv2_cli` as a root option.
* ``RV_CONFIG`` - Optional path to the specific Raster Vision Configuration file. These configurations will override  configurations that exist in configurations files in the default locations, but will not cause those configurations to be ignored.
* ``RV_CONFIG_DIR`` - Optional path to the directory that contains Raster Vision configuration. Defaults to ``${HOME}/.rastervision``

.. _rv2_running on gpu:

Running on a machine with GPUs
------------------------------

If you would like to run Raster Vision in a Docker container with GPUs - e.g. if you have your own GPU machine or you spun up a GPU-enabled machine on a cloud provider like a p3.2xlarge on AWS - you'll need to check some things so that the Docker container can utilize the GPUs.

Here are some (slightly out of date, but still useful) `instructions <https://github.com/agroimpacts/geog287387/blob/master/materials/tutorials/ubuntu-deeplearning-ami-raster-vision.md>`_ written by a community member on setting up an AWS account and a GPU-enabled EC2 instance to run Raster Vision.

Install nvidia-docker
~~~~~~~~~~~~~~~~~~~~~

You'll need to install the `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ runtime on your system. Follow their `Quickstart <https://github.com/NVIDIA/nvidia-docker#quickstart>`_ and installation instructions. Make sure that your GPU is supported by NVIDIA Docker - if not you might need to find another way to have your Docker container communicate with the GPU. If you figure out how to support more GPUs, please let us know so we can add the steps to this documentation!

Use the nvidia-docker runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When running your Docker container, be sure to include the ``--runtime=nvidia`` option, e.g.

.. code-block:: console

   > docker run --runtime=nvidia --rm -it quay.io/azavea/raster-vision:pytorch-0.12 /bin/bash

Ensure your setup sees the GPUS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend you ensure that the GPUs are actually enabled. If you don't, you may run a training job that you think is using the GPU and isn't, and runs very slowly.

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

.. _rv2_aws batch setup:

Setting up AWS Batch
--------------------

To run Raster Vision using AWS Batch, you'll need to setup your AWS account with a specific set of Batch resources, which you can do using the CloudFormation template in the `Raster Vision AWS Batch repository <https://github.com/azavea/raster-vision-aws>`_.

.. _rv2_aws batch config:

AWS Batch Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After creating the resources on AWS, set the following configuration in your :ref:`rv2_raster vision config`.
Check the AWS Batch console to see the names of the resources that were created, as they vary depending on how CloudFormation was configured.

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
   For more information about how Raster Vision uses AWS Batch, see the section: :ref:`rv2_aws batch`.
