Setup
=====

.. _install raster vision:

Installing Raster Vision
------------------------

.. currentmodule:: rastervision

You can get the library directly from PyPI:

.. code-block:: console

    > pip install rastervision

.. note:: Raster Vision requires Python 3 or later.

Troubleshooting macOS Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter problems running ``pip install rastervision`` on macOS, you may have to manually install Cython and pyproj.

To circumvent a problem installing pyproj with Python 3.7, you may also have to install that library using ``git+https``:

.. code-block:: console

   > pip install cython
   > pip install git+https://github.com/jswhit/pyproj.git
   > pip install rastervision

Using AWS, Tensorflow, and/or Keras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you'd like to use AWS, Tensorflow and/or Keras with Raster Vision, you can include any of these extras:

.. code-block:: console

    > pip install rastervision[aws,tensorflow,tensorflow-gpu]

If you'd like to use Raster Vision with `Tensorflow Object Detection <https://github.com/tensorflow/models/tree/master/research/object_detection>`_ or `TensorFlow DeepLab <https://github.com/tensorflow/models/tree/master/research/deeplab>`_, you'll need to follow the instructions in their documentation about how to install, or look at our Dockerfile to see an example of setting this up.

.. note:: You must install Tensorflow Object Detection and Deep Lab from `Azavea's fork <https://github.com/azavea/models/tree/AZ-v1.11-RV-v0.8.0>`_ of the models repository, since it contains some necessary changes that have not yet been merged back upstream.

.. note:: The usage of :ref:`docker containers` is recommended, as it provides a consistent environment for running Raster Vision.

If you have Docker installed, simply run the published container according to the instructions in :ref:`docker containers`

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
^^^^^^^^

Profiles allow you to specify profile names from the command line or environment variables
to determine which settings to use. The configuration file used will be named the same as the
profile: if you had two profiles (the ``default`` and one named ``myprofile``), your
``${HOME}/.rastervision`` would look like this:

.. code-block:: console

   > ls ~/.rastervision
   default    myprofile

See the root options of the :ref:`cli` for the option to set the profile.

Configuration File Sections
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _rv config section:

RV
~~~

.. code-block:: ini

   [RV]
   model_defaults_uri = ""

* ``model_defaults_uri`` - Specifies the URI of the :ref:`model defaults` JSON. Leave this option out to use the Raster Vision supplied model defaults.

.. _plugins config section:

PLUGINS
~~~~~~~

.. code-block:: ini

   [PLUGINS]
   files=[]
   modules=[]

* ``files`` - Optional list of Python file URIs to gather plugins from. Must be a JSON-parsable array of values, e.g. ``["analyzers.py","backends.py"]``.
* ``modules`` - Optional list of modules to load plugins from. Must be a JSON-parsable array of values, e.g. ``["rvplugins.analyzer","rvplugins.backend"]``.

See :ref:`plugins` for more information about the Plugin architecture.

Other Sections
~~~~~~~~~~~~~~

Other configurations are documented elsewhere:

* :ref:`aws batch config section`

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

Any INI file option can also be stated in the environment. Just prepend the section name to the setting name, e.g. ``RV_MODEL_DEFAULTS_URI``.

In addition to those environment variables that match the INI file values, there are the following environment variable options:

* ``TMPDIR`` - Setting this environment variable will cause all temporary directories to be created inside this folder. This is useful, for example, when you have a Docker container setup that mounts large network storage into a specific directory inside the Docker container. The tmp_dir can also be set on :ref:`cli` as a root option.
* ``RV_CONFIG`` - Optional path to the specific Raster Vision Configuration file. These configurations will override  configurations that exist in configurations files in the default locations, but will not cause those configurations to be ignored.
* ``RV_CONFIG_DIR`` - Optional path to the directory that contains Raster Vision configuration. Defaults to ``${HOME}/.rastervision``

.. _docker containers:

Docker Containers
-----------------

Using the Docker containers published for Raster Vision allows
you to use a fully set up environment. We have tested this with Docker 18, although you may be able to use a lower version.

Docker containers are published to `quay.io/azavea/raster-vision <https://quay.io/repository/azavea/raster-vision>`_. To run the raster vision container for the latest release, run:

.. code-block:: console

   > docker run --rm -it quay.io/azavea/raster-vision:cpu-0.8 /bin/bash

You'll likely need to load up volumes and expose ports to make this container fully useful; see the `docker/console <https://github.com/azavea/raster-vision/blob/0.8/docker/console>`_ script for an example usage.

We publish containers set up for both CPU-only running and GPU-running, and tag each container as appropriate. So you can also pull down the ``quay.io/azavea/raster-vision:gpu-0.8`` image, as well as ``quay.io/azavea/raster-vision:cpu-latest`` and ``quay.io/azavea/raster-vision:gpu-latest``.

You can also base your own Dockerfiles off the Raster Vision container to use with your own codebase. See the Dockerfiles in the `Raster Vision Examples <https://github.com/azavea/raster-vision-examples>`_ repository.

.. _running on gpu:

Running on a machine with GPUs
------------------------------

If you are running Raster Vision in a Docker container with GPUs - e.g. if you have your own GPU machine or you spun up a GPU-enabled machine on a cloud provider like a p3.2xlarge on AWS - you'll need to make sure of a couple of things so that the Docker container is able to utilize the GPUs.

Install nvidia-docker
^^^^^^^^^^^^^^^^^^^^^

You'll need to install the `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ runtime on your system. Follow their `quickstart <https://github.com/NVIDIA/nvidia-docker#quickstart>`_ and installation instructions. Make sure that your GPU is supported by NVIDIA Docker - if not you might need to find another way to have your Docker container communicate with the GPU. If you figure out how to support more GPUs, please let us know so we can add the steps to this documentation!

Use the nvidia-docker runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running your Docker container, be sure to include the ``--runtime=nvidia`` option, e.g.

.. code-block:: console

   > docker run --runtime=nvidia --rm -it quay.io/azavea/raster-vision:gpu-0.8 /bin/bash

Ensure your setup sees the GPUS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. _aws batch setup:

Setting up AWS Batch
--------------------

If you want to run code against AWS, you'll need a specific Raster Vision AWS Batch setup on your account, which you can accomplish by following the instructions found in the `CloudFormation-based Raster Vision for AWS Batch setup repository <https://github.com/azavea/raster-vision-batch>`_.

.. _aws batch config section:

AWS Batch Configuration Section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the appropriate configuration in your :ref:`raster vision config`:

.. code:: ini

   [AWS_BATCH]
   job_queue=RasterVisionGpuJobQueue
   cpu_job_queue=RasterVisionCpuJobQueue
   job_definition=RasterVisionHostedGpuJobDefinition
   cpu_job_definition=RasterVisionHostedCpuJobDefinition
   attempts=1


* ``job_queue`` - Job Queue to submit GPU Batch jobs to.
* ``cpu_job_queue`` - Job Queue to submit CPU-only jobs to.
* ``job_definition`` - The Job Definition that defines the Batch jobs to run.
* ``cpu_job_definition`` - The Job Definition that defines the CPU Batch jobs to run.
* ``attempts`` - Optional number of attempts to retry failed jobs.

The names of the resources in the example profile above were generated by running the `Raster Vision for AWS Batch setup repository <https://github.com/azavea/raster-vision-cloudformation>`_. The exact names may vary depending on how the CloudFormation stack is configured. Check the AWS Batch console to see the names of the resources that were created.

.. seealso::
   For more information about how Raster Vision uses AWS Batch, see the section: :ref:`aws batch`.
