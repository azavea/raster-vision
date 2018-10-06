Command  Line Interface
=======================

For usage information, type

.. code:: shell

   > rastervision --help

for usage information for a specific command, type --help after  the command, e.g.

.. code:: shell

   > rastervision run --help


Commands
--------

run
^^^

The run command

Use ``-a`` to pass arguments into the experiment methods; many of which take a root_uri, which is where Raster Vision will store all the output of the experiment. If you forget to supply this, Raster Vision will remind you.

Using the ``-n`` or ``--dry-run`` flag is useful to see what you're about to run before you run it. Combine this with the verbose flag for different levels of output:

.. code:: shell

   > rastervision run spacenet.chip_classification -a root_uri s3://example/ --dry_run
   > rastervision -v run spacenet.chip_classification -a root_uri s3://example/ --dry_run
   > rastervision -vv run spacenet.chip_classification -a root_uri s3://example/ --dry_run


Use ``-x`` to avoid checking if files exist, which can take a long time for large experiments. This is useful to do the first run, but if you haven't changed anything about the experiment and are sure the files are there, it's often nice to skip that step.

ls
^^^

The ls command
