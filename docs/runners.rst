.. _runners:

Running Pipelines
===================

Running pipelines in Raster Vision is done using the ``rastervision`` :ref:`run cli command` command. This generates a pipeline configuration, serializes it, and then uses a runner to actually execute the commands, locally or remotely.

.. seealso:: :ref:`pipeline package` explains more of the details of how ``Pipelines`` are implemented.

Running locally
---------------

local
^^^^^

A ``rastervision run local ...`` command will use the ``LocalRunner``, which
builds a ``Makefile`` based on the pipeline and executes it on the host machine. This will run multiple pipelines in parallel, as well as splittable commands in parallel, by spawning new processes for each command, where each process runs ``rastervision run_command ...``.

inprocess
^^^^^^^^^^

For debugging purposes, using ``rastervision run inprocess`` will run everything sequentially within a single process.

.. _aws batch:

Running remotely
-----------------

batch
^^^^^^

Running ``rastervision run batch ...`` will submit a DAG (directed acyclic graph) of jobs to be run on AWS Batch, which will increase the instance count to meet the workload with low-cost spot instances, and terminate the instances when the queue of commands is finished. It can also run some commands on CPU instances (like ``chip``), and others on GPU (like ``train``), and will run multiple experiments in parallel, as well as splittable commands in parallel.

The ``AWSBatchRunner`` executes each command by submitting a job to Batch, which executes the ``rastervision run_command``
inside the Docker image configured in the Batch job definition.
Commands that are dependent on an upstream command are submitted as a job after the upstream
command's job, with the ``jobId`` of the upstream command job as the parent ``jobId``. This way Batch knows to wait to execute each command until all upstream commands are finished
executing, and will fail the command if any upstream commands fail.

If you are running on AWS Batch or any other remote runner, you will not be able to use your local file system to store any of the data associated with an experiment.

.. note::
   To run on AWS Batch, you'll need the proper setup. See :ref:`aws batch setup` for instructions.

.. _parallelizing commands:

Running Commands in Parallel
----------------------------

Raster Vision can run certain commands in parallel, such as the :ref:`chip command` and :ref:`predict command` commands. These commands are designated as ``split_commands`` in the corresponding ``Pipeline`` class. To run split commands in parallel, use the ``--split`` option to the :ref:`run cli command` CLI command.

Splittable commands can be run in parallel, with each instance doing its share of the workload. For instance, using ``--splits 5`` on a ``CHIP`` command over
50 training scenes and 25 validation scenes will result in 5 CHIP commands running in parallel, that will each create chips for 15 scenes.

The command DAG that is given to the runner is constructed such that each split command can be run in parallel if the runner supports parallelization, and that any command that is dependent on the output of the split command will be dependent on each of the splits. So that means, in the above example,
a ``TRAIN`` command, which was dependent on a single ``CHIP`` command pre-split, will be dependent each of the 5 individual ``CHIP`` commands after the split.

Each runner will handle parallelization differently. For instance, the local runner will run each
of the splits simultaneously, so be sure the split number is in relation to the number of CPUs available.
The AWS Batch runner will use `array jobs <https://docs.aws.amazon.com/batch/latest/userguide/array_jobs.html>`_ to run commands in parallel, and the Batch Compute Environment will determine how many resources are available to run jobs simultaneously.
