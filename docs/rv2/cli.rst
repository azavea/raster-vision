.. _rv2_cli:

Command Line Interface
=======================

.. currentmodule:: rastervision2

The Raster Vision command line utility, ``rastervision2``, is installed with a ``pip install`` of
``rastervision``, which is installed by default in the :ref:`docker containers`.
It has subcommands, with some top level options:

.. code-block:: terminal

   > rastervision2 --help

    Usage: rastervision2 [OPTIONS] COMMAND [ARGS]...

    The main click command.

    Sets the profile, verbosity, and tmp_dir in RVConfig.

    Options:
    -p, --profile TEXT  Sets the configuration profile name to use.
    -v, --verbose       Increment the verbosity level.
    --tmpdir TEXT       Root of temporary directories to use.
    --help              Show this message and exit.

    Commands:
    predict      Use a model bundle to predict on new images.
    run          Run sequence of commands within pipeline(s).
    run_command  Run an individual command within a pipeline.

Commands
--------

.. _run cli command:

run
^^^

Run is the main interface into running pipelines.

.. code-block:: terminal

    > rastervision2 run --help

    Usage: rastervision2 run [OPTIONS] RUNNER CFG_MODULE [COMMANDS]...

    Run COMMANDS within pipelines in CFG_MODULE using RUNNER.

    RUNNER: name of the Runner to use

    CFG_MODULE: the module with `get_configs` function that returns
    PipelineConfigs. This can either be a Python module path or a local path
    to a .py file.

    COMMANDS: space separated sequence of commands to run within pipeline. The
    order in which to run them is based on the Pipeline.commands attribute. If
    this is omitted, all commands will be run.

    Options:
    -a, --arg KEY VALUE   Arguments to pass to get_config function
    -s, --splits INTEGER  Number of processes to run in parallel for splittable
                          commands
    --help                Show this message and exit.

Some specific parameters to call out:

-\\-splits
~~~~~~~~~~

Use ``-s N`` or ``--splits N``, where ``N`` is the number of splits to create, to parallelize commands that can be split into parallelizable chunks. See :ref:`parallelizing commands` for more information.

.. _predict cli command:

run_command
^^^^^^^^^^^

The ``run_command`` is used to run a specific command from a serialized pipeline configuration.
This is likely only useful to people writing :ref:`pipeline runner` that want to run
commands remotely from serialzed ``PipelineConfig`` JSON.

.. code-block:: terminal

    > rastervision2 run_command --help

    Usage: rastervision2 run_command [OPTIONS] CFG_JSON_URI COMMAND

    Run a single COMMAND using a serialized PipelineConfig in CFG_JSON_URI.

    Options:
    --split-ind INTEGER   The process index of a split command
    --num-splits INTEGER  The number of processes to use for running splittable
                            commands
    --runner TEXT         Name of runner to use
    --help                Show this message and exit.

predict
^^^^^^^

Use ``predict`` to make predictions on new imagery given a :ref:`model bundle`.

.. code-block:: terminal

    > rastervision2 predict --help

    Usage: rastervision2 predict [OPTIONS] MODEL_BUNDLE IMAGE_URI OUTPUT_URI

    Make predictions on the images at IMAGE_URI using MODEL_BUNDLE and store
    the prediction output at OUTPUT_URI.

    Options:
    -a, --update-stats    Run an analysis on this individual image, as opposed
                            to using any analysis like statistics that exist in
                            the prediction package
    --channel-order TEXT  List of indices comprising channel_order. Example: 2 1
                            0
    --help                Show this message and exit.
