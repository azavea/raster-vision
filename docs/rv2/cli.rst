.. _rv2_cli:

Command Line Interface
=======================

.. currentmodule:: rastervision

The Raster Vision command line utility, ``rastervision``, is installed with a ``pip install`` of
``rastervision``, which is installed by default in the :ref:`docker containers`.
It has subcommands, with some top level options:

.. code-block:: terminal

   > rastervision --help
    Usage: python -m rastervision [OPTIONS] COMMAND [ARGS]...

    Options:
      -p, --profile TEXT  Sets the configuration profile name to use.
      -v, --verbose       Sets the output to  be verbose.
      --help              Show this message and exit.

    Commands:
      ls           Print out a list of Experiment IDs.
      predict      Make predictions using a predict package.
      run          Run Raster Vision commands against Experiments.
      run_command  Run a command from configuration file.


Commands
--------

.. _run cli command:

run
^^^

Run is the main interface into running ``ExperimentSet`` workflows.

.. code-block:: terminal

    > rastervision run --help
    Usage: python -m rastervision run [OPTIONS] RUNNER [COMMANDS]...

    Run Raster Vision commands from experiments, using the experiment runner
    named RUNNER.

    Options:
      -e, --experiment_module TEXT  Name of an importable module to look for
                                    experiment sets in. If not supplied,
                                    experiments will be loaded from __main__
      -p, --path PATTERN            Path of file containing ExprimentSet to run.
      -n, --dry-run                 Execute a dry run, which will print out
                                    information about the commands to be run, but
                                    will not actually run the commands
      -x, --skip-file-check         Skip the step that verifies that file exist.
      -a, --arg KEY VALUE           Pass a parameter to the experiments if the
                                    method parameter list takes in a parameter
                                    with that key. Multiple args can be supplied
      --prefix PREFIX               Prefix for methods containing experiments.
                                    (default: "exp_")
      -m, --method PATTERN          Pattern to match method names to run.
      -f, --filter PATTERN          Pattern to match experiment names to run.
      -r, --rerun                   Rerun commands, regardless if their output
                                    files already exist.
      --tempdir TEXT                Temporary directory to use for this run.
      -s, --splits INTEGER          The number of processes to attempt to split
                                    each stage into.
      --help                        Show this message and exit.

Some specific parameters to call out:

-\\-arg
~~~~~~~~~~~

Use ``-a`` to pass arguments into the experiment methods; many of which take a ``root_uri`` which is where Raster Vision will store all the output of the experiment. If you forget to supply an argument, Raster Vision will remind you.

-\\-dry-run
~~~~~~~~~~~

Using the ``-n`` or ``--dry-run`` flag is useful to see what you're about to run before you run it. Combine this with the verbose flag for different levels of output:

.. code:: shell

   > rastervision run spacenet.chip_classification -a root_uri s3://example/ --dry_run
   > rastervision -v run spacenet.chip_classification -a root_uri s3://example/ --dry_run
   > rastervision -vv run spacenet.chip_classification -a root_uri s3://example/ --dry_run

-\\-skip-file-check
~~~~~~~~~~~~~~~~~~~

Use ``--skip-file-check`` or ``-x`` to avoid checking if files exist, which can take a long time for large experiments. This is useful to do the first run, but if you haven't changed anything about the experiment and are sure the files are there, it's often nice to skip that step.

.. _run split option:

-\\-splits
~~~~~~~~~~

Use ``-s N`` or ``--splits N``, where ``N`` is the number of splits to create, to parallelize commands that can be split into parallelizable chunks. See :ref:`parallelizing commands` for more information.

.. _predict cli command:

predict
^^^^^^^

Use ``predict`` to make predictions on new imagery given a :ref:`predict package`.

.. code-block:: terminal

    > rastervision predict --help
    Usage: python -m rastervision predict [OPTIONS] PREDICT_PACKAGE IMAGE_URI
                                          OUTPUT_URI

      Make predictions on the image at IMAGE_URI using PREDICT_PACKAGE and store
      the prediciton output at OUTPUT_URI.

    Options:
      -a, --update-stats    Run an analysis on this individual image, as opposed
                            to using any analysis like statistics that exist in
                            the prediction package
      --channel-order TEXT  List of indices comprising channel_order. Example: 2 1
                            0
      --export-config PATH  Exports the configuration to the given output file.
      --help                Show this message and exit.

ls
^^^

The ``ls`` command very simply lists the IDs of experiments in the given module or file.
This functionality is likely to expand to give more information about expriments discovered
in a project in later versions.

.. code-block:: terminal

    > rastervision ls --help
    Usage: python -m rastervision ls [OPTIONS]

      Print out a list of Experiment IDs.

    Options:
      -e, --experiment-module TEXT  Name of an importable module to look for
                                    experiment sets in. If not supplied,
                                    experiments will be loaded from __main__
      -a, --arg KEY VALUE           Pass a parameter to the experiments if the
                                    method parameter list takes in a parameter
                                    with that key. Multiple args can be supplied
      --help                        Show this message and exit.

run_command
^^^^^^^^^^^

The ``run_command`` is used to run a specific command from a serialized command configuration.
This is likely only useful to people writing :ref:`experiment runner` that want to run
commands remotely from serialzed command JSON.

.. code-block:: terminal

    > rastervision run_command --help
    Usage: python -m rastervision run_command [OPTIONS] COMMAND_CONFIG_URI

    Run a command from a serialized command configuration at
    COMMAND_CONFIG_URI.

    Options:
    --tempdir TEXT
    --help          Show this message and exit.
