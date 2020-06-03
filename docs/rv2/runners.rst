Running Experiments
===================

Running experiments in Raster Vision is done using the ``rastervision`` :ref:`rv2_run cli command` command.
This looks in all the places stated by the command for :ref:`pipeline` objects and executes methods
to get a collection of :ref:`experiment` objects. These are fed into the ``ExperimentRunner`` that
is chosen as a command line argument, which then determines how the commands derived from the
experiments should be executed.

.. _experiment runner:

ExperimentRunners
-----------------

An ``ExperimentRunner`` takes a collection of :ref:`experiment` objects and executes commands
derived from those configurations. The commands it chooses to run are based on which commands
are requested from the user, which commands already have been run, and which commands are common
between ExperimentConfigs.

.. note:: Raster Vision considers two commands to be equal if their inputs, outputs and command types
          (e.g. rv.CHIP, rv.TRAIN, etc...) are the same. Raster Vision will avoid running multiple of
          the same command in one run with sameness defined in this way.

During the process of deriving commands from the ExperimentConfigs, each Config object in the
experiment has the chance to update itself for a specific command (using the ``update_for_command`` method), and report what its inputs
and outputs are (using the ``report_io`` method). This is an internal mechanism, so you won't have to dive too deeply into this
unless you are a contributor or a plugin author. However, it's good to know that this
is when some of the implicit values are set into the configuration. For instance,
the ``model_uri`` property can be set on a ``rv.BackendConfig`` by using the ``with_model_uri``
on the builder; however the more standard practice is to let Raster Vision set this property
during the ``update_for_command`` process described above, which it will do based on the
``root_uri`` of the ``ExperimentConfig`` as well as other factors.

The base ``ExperimentRunner`` class constructs a Directed Acyclic Graph (DAG) of the commands
based on which commands consume as input other command's outputs, and passes that off
to the implementation to be executed. The specific implementation will choose how to
actually execute each command.

When an ``ExperimentSet`` is executed by an ``ExperimentRunner``, it is first converted into a ``CommandDAG`` representing a DAG of commands. In this graph, there is a node for each command, and an edge from X to Y if X produces the input of Y. The commands are then executed according to a topological sort of the graph, so as to respect dependencies between commands.

Two optimizations are performed to eliminate duplicated computation. The first is to only execute commands whose outputs don't exist. The second is to eliminate duplicate nodes that are present when experiments partially overlap, like when an ``ExperimentSet`` is created with multiple experiments that generate the same chips:

.. image:: _static/commands-tree-workflow.png
    :align: center

Running locally
---------------

A ``rastervision run local ...`` command will use the ``LocalExperimentRunner``, which
builds a Makefile based on the DAG and then executes it on the host machine. This will run multiple experiments in parallel.

.. _aws batch:

Running on AWS Batch
--------------------

``rastervision run aws_batch ...`` will execute the commands on AWS Batch. This provides
a powerful mechanism for running Raster Vision experiment workflows. It allows
for queues of CPU and GPU instances to have 0 instances running when not in use. With the running of a
single command on your own machine, AWS Batch will increase the instance count to meet
the workload with low-cost spot instances, and terminate the instances when the queue
of commands is finished. It can also run some commands on CPU instances (like ``chip``), and others on GPU (like ``train``), and will run multiple experiments in parallel.

The ``AWSBatchExperimentRunner`` executes each command by submitting a job to Batch, which executes the ``rastervision run_command``
inside the Docker image configured in the Batch job definition.
Commands that are dependent on an upstream command are submitted as a job after the upstream
command's job, with the jobId of the upstream command job as the parent jobId. This way
AWS Batch knows to wait to execute each command until all upstream commands are finished
executing, and will fail the command if any upstream commands fail.

If you are running on AWS Batch or any other remote runner, you will not be able to use
your local file system to store any of the data associated with an experiment - this
includes plugin files.

.. note::
   To run on AWS Batch, you'll need the proper setup. See :ref:`aws batch setup` for instructions.

.. _parallelizing commands:

Running commands in Parallel
----------------------------

Raster Vision can run certain commands in parallel, such as the :ref:`chip command` and :ref:`predict command` commands. To do so, use the :ref:`run split option` option in the ``run`` command of the CLI.

Commands implement a ``split`` method on them, that either returns the original command if they
cannot be split, e.g. with training, or a sequence of commands that each do a subset of the work. For instance, using ``--splits 5`` on a ``CHIP`` command over
50 training scenes and 25 validation scenes will result in 5 CHIP commands, that can be run
in parallel, that will each create chips for 15 scenes.

The command DAG that is given to the experiment runner is constructed such that each split command
can be run in parallel if the runner supports parallelization, and that any command that is dependent on
the output of the split command will be dependent on each of the splits. So that means, in the above example,
a ``TRAIN`` command, which was dependent on a single ``CHIP`` command pre-split, will be dependent each of the
5 individual ``CHIP`` commands after the split.

Each runner will handle parallelization differently. For instance, the local runner will run each
of the splits simultaneously, so be sure the split number is in relation to the number of CPUs available.
The AWS Batch runner will submit jobs for each of the command splits, and the Batch Compute Environment will
dictate how  many resources are available to run Batch jobs simultaneously.
