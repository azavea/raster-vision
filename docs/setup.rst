.. _setup:

Setup
=====

.. _install raster vision:

Installing via pip
------------------------

.. currentmodule:: rastervision

You can directly install the library using ``pip`` (or ``pip3`` if you also have Python 2 installed).

.. code-block:: console

   > pip install rastervision=={{ version }}

This has been shown to work in the following environment. Variations on this environment may or may not work.

* Ubuntu Linux 20.04
* Python 3.9
* CUDA 11.6 and NVIDIA Driver 510.47.03 (for GPU support)

Raster Vision also runs on macOS version 12.1, except that the ``num_workers`` for the ``DataLoader`` will need to be set to 0 due to an issue with mulitprocessing on Macs with Python >= 3.8. It will also be necessary to install GDAL 3.5.2 prior to installing Raster Vision, which isn't necessary on Linux.

.. warning::

    Raster Vision has not been tested with Windows, and will probably run into problems.

An alternative approach for running Raster Vision is to use the provided :ref:`docker images <Docker images>` which encapsulate a complete environment that is known to work.

Install individual pip packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raster Vision is comprised of a required ``rastervision.pipeline`` package, and a number of optional plugin packages, as described in :ref:`codebase overview`. Each of these packages have their own dependencies, and can be installed individually. Running the following command:

.. code-block:: console

    > pip install rastervision=={{ version }}

is equivalent to running the following sequence of commands:

.. code-block:: console

    > pip install rastervision_pipeline=={{ version }}
    > pip install rastervision_aws_s3=={{ version }}
    > pip install rastervision_aws_batch=={{ version }}
    > pip install rastervision_core=={{ version }}
    > pip install rastervision_pytorch_learner=={{ version }}
    > pip install rastervision_pytorch_backend=={{ version }}
    > pip install rastervision_gdal_vsi=={{ version }}

.. _docker images:

Docker Images
-----------------

Using the Docker images published for Raster Vision makes it easy to use a fully set up environment. We have tested this with Docker 20, although you may be able to use a lower version.

The images we publish include plugins and dependencies for using Raster Vision with PyTorch, AWS S3, and Batch. These are published to `quay.io/azavea/raster-vision <https://quay.io/repository/azavea/raster-vision>`_.  To run the container for the latest release, run:

.. code-block:: console

   > docker run --rm -it quay.io/azavea/raster-vision:pytorch-{{ version }} /bin/bash

There are also images with the `-latest` suffix for the latest commits on the ``master`` branch. You'll likely need to mount volumes and expose ports to make this container fully useful; see the `docker/run <https://github.com/azavea/raster-vision/blob/{{ version }}/docker/run>`_ script for an example usage.

You can also base your own Dockerfiles off the Raster Vision image to use with your own codebase. See :ref:`bootstrap` for more information.

Docker Scripts
~~~~~~~~~~~~~~

There are several scripts under `docker/ <https://github.com/azavea/raster-vision/tree/{{ version }}/docker>`_ in the Raster Vision repo that make it easier to build the Docker images from scratch, and run the container in various ways. These are useful if you are experimenting with changes to the Raster Vision source code, or writing :ref:`plugins <pipelines plugins>`.

After cloning the repo, you can build the Docker image using:

.. code-block:: console

    > docker/build

To build an image that can run natively on an ARM64 chip, pass the ``--arm64`` flag. This won't be necessary for most users, but if you have an ARM64 chip, like in a recent Macbook, this will speed things up greatly.

Before running the container, set an environment variable to a local directory in which to store data.

.. code-block:: console

    > export RASTER_VISION_DATA_DIR="/path/to/data"

To run a Bash console in the PyTorch Docker container use:

.. code-block:: console

    > docker/run

This will mount the ``$RASTER_VISION_DATA_DIR`` local directory to to ``/opt/data/`` inside the container.

.. warning::

    Users running under WSL2 in Windows will need to unset the ``NAME`` environment variable. For example, instead of
    ``docker/run``, you would run ``NAME='' docker/run``. By default, WSL2 sets a ``NAME`` variable that matches the network
    name of your computer. This environment variable collides with a variable in the ``docker/run`` script.

.. warning::

    If you have built an ARM64 image, you should pass the ``--arm64`` flag to ``docker/run``.

This script also has options for forwarding AWS credentials, and running Jupyter notebooks which can be seen below.

.. code-block:: console

    > docker/run --help

    Usage: run <options> <command>

    Run a console in a Raster Vision Docker image locally.
    By default, the raster-vision-pytorch image is used in the CPU runtime.

    Environment variables:
    RASTER_VISION_DATA_DIR (directory for storing data; mounted to /opt/data)
    RASTER_VISION_NOTEBOOK_DIR (optional directory for Jupyter notebooks; mounted to /opt/notebooks)
    AWS_PROFILE (optional AWS profile)

    Options:
    --aws forwards AWS credentials (sets AWS_PROFILE env var and mounts ~/.aws to /root/.aws)
    --tensorboard maps port 6006
    --name sets the name of the running container
    --jupyter forwards port 8888, mounts RASTER_VISION_NOTEBOOK_DIR to /opt/notebooks, and runs Jupyter
    --docs runs the docs server and forwards port 8000
    --debug forwards port 3000 for use with remote debugger
    --gpu use nvidia runtime
    --arm64 uses image built for arm64 architecture

    All arguments after above options are passed to 'docker run'.

.. _raster vision config:

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

Use the ``rastervision --profile`` option in the :ref:`cli` to set the profile.

Configuration File Sections
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _s3 config:

AWS_S3
^^^^^^

.. code-block:: ini

   [AWS_S3]
   requester_pays = False

* ``requester_pays`` - Set to True if you would like to allow using `requester pays <https://docs.aws.amazon.com/AmazonS3/latest/dev/RequesterPaysBuckets.html>`_ S3 buckets. The default value is False.

Other
^^^^^^

Other configurations are documented elsewhere:

* :ref:`aws batch setup`

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Any profile file option can also be stated in the environment. Just prepend the section name to the setting name, e.g. ``export AWS_S3_REQUESTER_PAYS="False"``.

In addition to those environment variables that match the INI file values, there are the following environment variable options:

* ``TMPDIR`` - Setting this environment variable will cause all temporary directories to be created inside this folder. This is useful, for example, when you have a Docker container setup that mounts large network storage into a specific directory inside the Docker container. The tmp_dir can also be set on :ref:`cli` as a root option.
* ``RV_CONFIG`` - Optional path to the specific Raster Vision Configuration file. These configurations will override  configurations that exist in configurations files in the default locations, but will not cause those configurations to be ignored.
* ``RV_CONFIG_DIR`` - Optional path to the directory that contains Raster Vision configuration. Defaults to ``${HOME}/.rastervision``

.. _running on gpu:

Running on a machine with GPUs
------------------------------

To run Raster Vision on a realistic dataset in a reasonable amount of time, it is necessary to use a machine with a GPU. Note that Raster Vision will use a GPU if it detects that one is available. If you don't own a machine with a GPU, it is possible to rent one by the minute using a cloud provider such as AWS.

Check that GPU is available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~

If you would like to run Raster Vision in a Docker container with GPUs, you'll need to check some things so that the Docker container can utilize the GPUs.

First, you'll need to install the `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ runtime on your system. Follow their `Quickstart <https://github.com/NVIDIA/nvidia-docker#quickstart>`_ and installation instructions. Make sure that your GPU is supported by NVIDIA Docker - if not you might need to find another way to have your Docker container communicate with the GPU. If you figure out how to support more GPUs, please let us know so we can add the steps to this documentation!

When running your Docker container, be sure to include the ``--runtime=nvidia`` option, e.g.

.. code-block:: console

   > docker run --runtime=nvidia --rm -it quay.io/azavea/raster-vision:pytorch-{{ version }} /bin/bash

or use the ``--gpu`` option with the ``docker/run`` script.

.. _aws ec2 setup:

Running on AWS EC2
~~~~~~~~~~~~~~~~~~~

The simplest way to run Raster Vision on an AWS GPU is by starting a GPU-enabled EC2 instance such as a p3.2xlarge using the `Deep Learning AMI <https://aws.amazon.com/machine-learning/amis/>`_. We have tested this using the "Deep Learning AMI GPU PyTorch 1.11.0 (Ubuntu 20.04)" with id ``ami-0c968d7ef8a4b0c34``. After SSH'ing into the instance, Raster Vision can be installed with ``pip``, and code can be transfered to this instance with a tool such as ``rsync``.

.. _aws batch setup:

Running on AWS Batch
~~~~~~~~~~~~~~~~~~~~

AWS Batch is a service that makes it easier to run Dockerized computation pipelines in the cloud. It starts and stops the appropriate instances automatically and runs jobs sequentially or in parallel according to the dependencies between them. To run Raster Vision using AWS Batch, you'll need to setup your AWS account with a specific set of Batch resources, which you can do using :ref:`cloudformation setup`. After creating the resources on AWS, set the following configuration in your :ref:`raster vision config`. Check the AWS Batch console to see the names of the resources that were created, as they vary depending on how CloudFormation was configured.

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
