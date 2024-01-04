.. _running on gpu:

Using GPUs
==========

To run Raster Vision on a realistic dataset in a reasonable amount of time, it is necessary to use a machine with one or more GPUs. Note that Raster Vision will automatically use all available GPUs.

If you don't own a machine with a GPU, it is possible to rent one by the minute using a cloud provider such as AWS. See :doc:`aws`.

Check that GPU is available
---------------------------

Regardless of how you are running Raster Vision, we recommend you ensure that the GPUs are actually enabled. If you don't, you may run a training job that you think is using the GPU and isn't, and runs very slowly.

One way to check this is to make sure PyTorch can see the GPU(s). To do this, open up a ``python`` console and run the following:

.. code-block:: console

    import torch
    torch.cuda.is_available()
    torch.cuda.device_count()

If you have `nvidia-smi <https://developer.nvidia.com/nvidia-system-management-interface>`_  installed, you can also use this command to inspect GPU utilization while the training job is running:

.. code-block:: console

    > watch -d -n 0.5 nvidia-smi

GPUs and Docker
---------------

If you would like to run Raster Vision in a Docker container with GPUs, you'll need to check some things so that the Docker container can utilize the GPUs.

First, you'll need to install the `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ runtime on your system. Follow their `Quickstart <https://github.com/NVIDIA/nvidia-docker#quickstart>`_ and installation instructions. Make sure that your GPU is supported by NVIDIA Docker - if not you might need to find another way to have your Docker container communicate with the GPU. If you figure out how to support more GPUs, please let us know so we can add the steps to this documentation!

When running your Docker container, be sure to include the ``--gpus=all`` option, e.g.

.. code-block:: console

   > docker run --gpus=all --rm -it quay.io/azavea/raster-vision:pytorch-{{ version }} /bin/bash

or use the ``--gpu`` option with the ``docker/run`` script.

.. _distributed:

Using multiple GPUs (distributed training)
------------------------------------------

Raster Vision supports distributed training (multi-node and multi-GPU) via `PyTorch DDP <https://pytorch.org/docs/master/notes/ddp.html>`_.

It can be used in the following ways:

- Run Raster Vision normally on a multi-GPU machine. Raster Vision will automatically detect the multiple GPU and use distributed training when ``Learner.train()`` is called. 
- Run Raster Vision using the `torchrun CLI command <https://pytorch.org/docs/stable/elastic/run.html>`_. For example, to run on a single machine with 4 GPUs:

  .. code-block:: console

     torchrun --standalone --nnodes=1 --nproc-per-node=4 --no-python \
        rastervision run local rastervision_pytorch_backend/rastervision/pytorch_backend/examples/tiny_spacenet.py

Other considerations
~~~~~~~~~~~~~~~~~~~~

- Config variables that may be :ref:`set via environment or RV config <raster vision config>` (also documented `here <https://raster-vision--2018.org.readthedocs.build/en/2018/api_reference/_generated/rastervision.pytorch_learner.learner.Learner.html#learner>`_):

  - ``RASTERVISION_USE_DDP``: ``YES`` by default. Set to ``NO`` to disable distributed training.
  - ``RASTERVISION_DDP_BACKEND``: ``nccl`` by default. This is the recommended backend for CUDA GPUs.
  - ``RASTERVISION_DDP_START_METHOD``: One of ``spawn``, ``fork``, or ``forkserver``. Passed to :func:`torch.multiprocessing.start_processes`. Default: ``spawn``.

    - ``spawn`` is what PyTorch documentation recommends (in fact, it doesn't even mention the alternatives), but it has the disadvantage that it requires everything to be pickleable, which rasterio dataset objects are not. This is also true for ``forkserver``, which needs to spawn a server process. However, ``fork`` does not have the same limitation.
    - If not ``fork``, we avoid building the dataset in the base process and instead delay it until the worker processes are created.
    - If ``fork`` or ``forkserver``, the CUDA runtime must not be initialized before the fork happens; otherwise, a ``RuntimeError: Cannot re-initialize CUDA in forked subprocess.`` error will be raised. We avoid this by not calling any ``torch.cuda`` functions or creating tensors on the GPU.
    
- To avoid having to re-download files for each process when building datasets, it is recommended to :meth:`manually specify a temporary directory <.RVConfig.set_tmp_dir_root>` (otherwise each process will use a separate randomly generated temporary directory). When a single temp directory is set, to avoid IO conflicts, Raster Vision first builds the datasets only in the master process (rank = 0) and only after in the other processes, so that they use the already downloaded files.
- A similar problem also occurs when downloading external models/losses, but in this case, the strategy of building on the master first does not work. The model apparently needs to be created by the same line of code on each process. Therefore, we need to download files separately for each process; we do this by modifying ``TORCH_HOME`` to ``$TORCH_HOME/<local rank>``. And only the master process copies the downloaded files to the training directory.
- Raster Vision will use all available GPUs by default. To override, set the ``WORLD_SIZE`` env var.
