Raster Vision Setup
===================

Requirements
------------

*  Docker 18+
*  awscli 1.15+

Scripts
-------

+---------------------+------------------------------------------+
| Name                | Description                              |
+=====================+==========================================+
| ``update``          | Build Docker containers                  |
+---------------------+------------------------------------------+
| ``run``             | Run container locally                    |
+---------------------+------------------------------------------+
| ``jupyter``         | Run Jupyter notebook in container with   |
|                     | mounted and notebook directory from      |
|                     | ``RASTER_VISION_NOTEBOOK_DIR``           |
+---------------------+------------------------------------------+
| ``cipublish``       | Publish Docker image to ECR              |
+---------------------+------------------------------------------+
| ``clean``           | Remove dangling containers               |
+---------------------+------------------------------------------+
| ``test``            | Run unit tests and lint on source code   |
+---------------------+------------------------------------------+
| ``setup_aws_batch`` | Setup AWS Batch                          |
+---------------------+------------------------------------------+


Initial setup
-------------

First, set the ``RASTER_VISION_DATA_DIR`` environment variable on your host machine. All data including datasets and results should be stored in a single directory outside of the repo. Also set the ``AWS_PfROFILE`` if you want to use AWS. Then build the CPU container using

.. code-block:: console

                ./scripts/update --cpu


Running locally on CPUs
-----------------------

Running the Docker container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can build the Docker container and then get into the Bash console for it using ``/scripts/run`` or ``./scripts/run --aws`` if you would like to forward AWS credentials.

Running an object detection workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the [object detection tutorial](object-detection.md).

Running a Jupyter notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can run a Juptyer notebook that has the data from ``RASTER_VISION_DATA_DIR`` mounted to ``/opt/data``
and ``RASTER_VISION_NOTEBOOK_DIR`` mounted to ``/opt/notebooks`` and set as the Juptyer notebook directory using

.. code-block:: console

                ./scripts/jupyter


Running remotely using AWS Batch
--------------------------------

In order to run Raster Vision on GPUs and in parallel, we use `AWS Batch <https://aws.amazon.com/batch/>`_

Publishing the container to ECR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The latest Docker image should be stored in ECR so that it can be used by Batch. To build and publish the container, run

.. code-block:: console

                ./scripts/cipublish

Creating / Updating the Batch AMI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run a GPU-enabled container on Batch, you need to use an AMI with the ECS agent installed as well as Nvidia drivers that are matched to the version of CUDA used in the Docker container. See `Vagrant/AMI docs <docs/vagrant-ami.md>`_ for more details.

Setup Batch resources
~~~~~~~~~~~~~~~~~~~~~

To setup the AWS Batch stack, which should only be done once per AWS account, run ``./scripts/setup_aws_batch``. You will need to change some values in ``deployment/batch`` for your individual setup.

Submitting a job
~~~~~~~~~~~~~~~~

To manually submit a job to Batch, use the ``batch_submit`` script inside the Docker container as follows.

.. code-block:: console

                python -m rastervision.utils.batch <branch_name> "<command_to_run>" --attempts <# of attempts>


The ``branch_name`` should be the name of the Git branch with the code to run. If you are testing a job to see if it might fail, you should run it with ``--attempts 1`` so that it won't be retried if it fails. After submitting a job, AWS Batch will start an EC2 instance, run the command inside a Docker container, and will shut it down when finished. You can also add the ``--gpu`` option to run it on a GPU enabled instance.
