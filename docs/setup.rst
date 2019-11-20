Setup
=====

.. _docker containers:

Docker Images
-----------------

Using the Docker images published for Raster Vision makes it easy to use a fully set up environment. We have tested this with Docker 18, although you may be able to use a lower version.

Docker images are published to `quay.io/azavea/raster-vision <https://quay.io/repository/azavea/raster-vision>`_. To run the container for the latest release, run:

.. code-block:: console

   > docker run --rm -it quay.io/azavea/raster-vision:pytorch-0.10 /bin/bash

You'll likely need to mount volumes and expose ports to make this container fully useful; see the `docker/run <https://github.com/azavea/raster-vision/blob/0.10/docker/run>`_ script for an example usage.

There are Raster Vision backends for PyTorch and Tensorflow -- the Tensorflow ones are being sunsetted.
We publish separate Docker images with the dependencies necessary for using the PyTorch and Tensorflow backends, and there are CPU and GPU variants for the Tensorflow images.
There are also images with the `-latest` suffix for the latest commits on the ``master`` branch. The available images include:

* ``quay.io/azavea/raster-vision:tf-gpu-0.10`` and ``quay.io/azavea/raster-vision:tf-gpu-latest``
* ``quay.io/azavea/raster-vision:tf-cpu-0.10`` and ``quay.io/azavea/raster-vision:tf-cpu-latest``
* ``quay.io/azavea/raster-vision:pytorch-0.10`` and ``quay.io/azavea/raster-vision:pytorch-latest``

You can also base your own Dockerfiles off the Raster Vision image to use with your own codebase. See the Dockerfiles in the `Raster Vision Examples <https://github.com/azavea/raster-vision-examples>`_ repository.

Docker Scripts
~~~~~~~~~~~~~~

There are several scripts under `docker/ <https://github.com/azavea/raster-vision/tree/0.10/docker>`_ in the Raster Vision repo that make it easier to build the Docker images from scratch, and run the container in various ways. These are useful if you are experimenting with changes to the Raster Vision source code.

After cloning the repo, you can build all the Docker images using:

.. code-block:: console

    > docker/build

Before running the container, set an environment variable to a local directory in which to store data.

.. code-block:: console

    > export RASTER_VISION_DATA_DIR="/path/to/data"

To run a Bash console in the PyTorch Docker container use:

.. code-block:: console

    > docker/run

This will mount the ``$RASTER_VISION_DATA_DIR`` local directory to to ``/opt/data/`` inside the container.

This script also has options for forwarding AWS credentials, running Jupyter notebooks, and switching between different images, which can be seen below.

Remember to use the correct image for the backend you are using!

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
    --tf-gpu use raster-vision-examples-tf-gpu image and nvidia runtime
    --tf-cpu use raster-vision-examples-tf-cpu image
    --pytorch-gpu use raster-vision-examples-pytorch image and nvidia runtime

    All arguments after above options are passed to 'docker run'.

.. _install raster vision:

Installing via pip
------------------------

.. currentmodule:: rastervision

Rather than running Raster Vision from inside a Docker container, you can directly install the library using ``pip``. However, we recommend using the Docker images since it can be difficult to install some of the dependencies.

.. code-block:: console

   > pip install rastervision==0.10.0

.. note:: Raster Vision requires Python 3 or later. Use ``pip3 install rastervision==0.10.0`` if you have more than one version of Python installed.

Troubleshooting macOS Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter problems running ``pip install rastervision==0.10.0`` on macOS, you may have to manually install Cython and pyproj.

To circumvent a problem installing pyproj with Python 3.7, you may also have to install that library using ``git+https``:

.. code-block:: console

  > pip install cython
  > pip install git+https://github.com/jswhit/pyproj.git@e56e879438f0a1688b89b33228ebda0f0d885c19
  > pip install rastervision==0.10.0

Using AWS, Tensorflow, and/or Keras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you'd like to use AWS, PyTorch, Tensorflow and/or Keras with Raster Vision, you can include any of these extras:

.. code-block:: console

   > pip install rastervision[aws,pytorch,tensorflow-cpu,tensorflow-gpu]==0.10.0

If you'd like to use Raster Vision with `Tensorflow Object Detection <https://github.com/tensorflow/models/tree/master/research/object_detection>`_ or `TensorFlow DeepLab <https://github.com/tensorflow/models/tree/master/research/deeplab>`_, you'll need to install these from `Azavea's fork <https://github.com/azavea/models/tree/AZ-v1.11-RV-v0.8.0>`_ of the models repository, since it contains some necessary changes that have not yet been merged back upstream.

You will also need to install `Tippecanoe <https://github.com/mapbox/tippecanoe>`_ if you would like to do vector tile processing. For an example of setting these up, see the various `Dockerfiles  <https://github.com/azavea/raster-vision/blob/0.10/>`_.

.. _raster vision config:

Raster Vision Configuration
---------------------------

Raster Vision is configured via the `everett <https://everett.readthedocs.io/en/latest/index.html>`_ library.

Raster Vision will look for configuration in the following locations, in this order:

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

.. _rv config section:

RV
^^

.. code-block:: ini

   [RV]
   model_defaults_uri = ""

* ``model_defaults_uri`` - Specifies the URI of the :ref:`model defaults` JSON. Leave this option out to use the Raster Vision supplied model defaults.

.. _s3 config section:

AWS_S3
^^^^^^

.. code-block:: ini

   [AWS_S3]
   requester_pays = False

* ``requester_pays`` - Set to True if you would like to allow using `requester pays <https://docs.aws.amazon.com/AmazonS3/latest/dev/RequesterPaysBuckets.html>`_ S3 buckets. The default value is False.

.. _plugins config section:

PLUGINS
^^^^^^^

.. code-block:: ini

   [PLUGINS]
   files=analyzers.py,backends.py
   modules=rvplugins.analyzer,rvplugins.backend

* ``files`` - Optional list of Python file URIs to gather plugins from as a comma-separated list of values, e.g. ``analyzers.py,backends.py``.
* ``modules`` - Optional list of modules to load plugins from as a comma-separated list of values, e.g. ``rvplugins.analyzer,rvplugins.backend``.

See :ref:`plugins` for more information about the Plugin architecture.

Other Sections
~~~~~~~~~~~~~~

Other configurations are documented elsewhere:

* :ref:`aws batch config section`

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Any INI file option can also be stated in the environment. Just prepend the section name to the setting name, e.g. ``RV_MODEL_DEFAULTS_URI``.

In addition to those environment variables that match the INI file values, there are the following environment variable options:

* ``TMPDIR`` - Setting this environment variable will cause all temporary directories to be created inside this folder. This is useful, for example, when you have a Docker container setup that mounts large network storage into a specific directory inside the Docker container. The tmp_dir can also be set on :ref:`cli` as a root option.
* ``RV_CONFIG`` - Optional path to the specific Raster Vision Configuration file. These configurations will override  configurations that exist in configurations files in the default locations, but will not cause those configurations to be ignored.
* ``RV_CONFIG_DIR`` - Optional path to the directory that contains Raster Vision configuration. Defaults to ``${HOME}/.rastervision``


.. _running on gpu:

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

   > docker run --runtime=nvidia --rm -it quay.io/azavea/raster-vision:pytorch-0.10 /bin/bash

Ensure your setup sees the GPUS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend you ensure that the GPUs are actually enabled. If you don't, you may run a training job that you think is using the GPU and isn't, and runs very slowly.

One way to check this is to make sure TensorFlow can see the GPU(s). To do this, open up an ipython console and initialize TensorFlow:

.. code-block:: console

   > ipython
   In [1]: import tensorflow as tf
   In [2]: sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

This should print out console output that looks something like:

.. code-block:: console

    .../gpu/gpu_device.cc:1405] Found device 0 with properties: name: GeForce GTX

If you have `nvidia-smi <https://developer.nvidia.com/nvidia-system-management-interface>`_  installed, you can also use this command to inspect GPU utilization while the training job is running:

.. code-block:: console

    > watch -d -n 0.5 nvidia-smi

Multi GPU systems
~~~~~~~~~~~~~~~~~
The chip classification task automatically makes use of multiple GPU's.

.. _aws batch setup:

Setting up AWS Batch
--------------------

To run Raster Vision using AWS Batch, you'll need to setup your AWS account with a specific set of Batch resources, which you can do using the CloudFormation template in the `Raster Vision AWS Batch repository <https://github.com/azavea/raster-vision-aws>`_.

.. _aws batch config section:

AWS Batch Configuration Section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After creating the resources on AWS, set the corresponding configuration in your :ref:`raster vision config`:

.. code:: ini

    [AWS_BATCH]
    job_queue=RasterVisionGpuJobQueue
    job_definition=RasterVisionHostedPyTorchGpuJobDefinition
    cpu_job_queue=RasterVisionCpuJobQueue
    cpu_job_definition=RasterVisionHostedPyTorchCpuJobDefinition
    attempts=5

* ``job_queue`` - Job Queue to submit GPU Batch jobs to.
* ``cpu_job_queue`` - Job Queue to submit CPU-only jobs to.
* ``job_definition`` - The Job Definition that defines the Batch jobs to run on GPU.
* ``cpu_job_definition`` - The Job Definition that defines the Batch jobs to run on CPU (which might be the same as the ``job_definition``)
* ``attempts`` - Optional number of attempts to retry failed jobs.

Check the AWS Batch console to see the names of the resources that were created, as they vary depending on how CloudFormation was configured.

If you would like the ability to switch between PyTorch and Tensorflow-based jobs, you should create separate Raster Vision profiles for each of the two sets of resources.

.. seealso::
   For more information about how Raster Vision uses AWS Batch, see the section: :ref:`aws batch`.
