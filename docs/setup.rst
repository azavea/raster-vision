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
| ``clean``           | Remove dangling containers               |
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
