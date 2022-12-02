.. _setup:

Installation
============

.. toctree::
    :maxdepth: 2
    :hidden:

    self
    configure
    gpu

.. _install raster vision:

Installing via pip
------------------

.. currentmodule:: rastervision

You can directly install the library using ``pip`` (or ``pip3`` if you also have Python 2 installed).

.. code-block:: console

   > pip install rastervision=={{ version }}

This has been shown to work in the following environment. Variations on this environment may or may not work.

* Ubuntu Linux 20.04
* Python 3.9
* CUDA 11.6 and NVIDIA Driver 510.47.03 (for GPU support)

Raster Vision also runs on macOS version 12.1, except that the ``num_workers`` for the :class:`~torch.utils.data.DataLoader` will need to be set to 0 due to an issue with mulitprocessing on Macs with Python >= 3.8. It will also be necessary to install GDAL 3.5.2 prior to installing Raster Vision, which isn't necessary on Linux.

.. warning::

    Raster Vision has not been tested with Windows, and will probably run into problems.

An alternative approach for running Raster Vision is to use the provided :ref:`docker images <Docker images>` which encapsulate a complete environment that is known to work.

Install individual pip packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raster Vision is comprised of a required :mod:`rastervision.pipeline` package, and a number of optional plugin packages, as described in :ref:`codebase overview`. Each of these packages have their own dependencies, and can be installed individually. Running the following command:

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
-------------

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
