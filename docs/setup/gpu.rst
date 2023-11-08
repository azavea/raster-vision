.. _running on gpu:

Using GPUs
==========

To run Raster Vision on a realistic dataset in a reasonable amount of time, it is necessary to use a machine with a GPU. Note that Raster Vision will use a GPU if it detects that one is available. 

If you don't own a machine with a GPU, it is possible to rent one by the minute using a cloud provider such as AWS. See :doc:`aws`.

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
