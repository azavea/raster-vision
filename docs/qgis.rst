QGIS Plugin
===========

TKTK

Installing
----------

TKTK

.. note:: The QGIS Plugin is in the process of being published to the `official QGIS plugin repository <https://plugins.qgis.org//>`_. This documentation will be updated once the release is approved.

Configuration
-------------

Configure the plugin with a working directory and an AWS profile (option, it will use your default profile if none is specified).  If the files live on S3, this plugin will download files as necessary to your local working directory. If the file already exists in the working directory, the plugin will check the timestamps and overwrite the local file if the file on S3 is newer.

Using with AWS
^^^^^^^^^^^^^^

You'll need to set your AWS_PROFILE in the QGIS environment if you're not using the defaul profile.

Using with docker
^^^^^^^^^^^^^^^^^

To run predict through docker, make sure that the docker command is on the `PATH` environment variable used  by docker.

Loading Experiment data
-----------------------

TKTK

Making Predicts
---------------

TKTK

Style Profiles
--------------

Set up style profiles so that when you load an experiment, layers are automatically styled with given SLDs.

The best way to do this is to styl each of the types of layers you want after first loading an experiment. Export an SLD of style for each layer by using the `Style` -> `Save Style` command in the `Symbology` section of the layer properties. Then, create a style profile for that experiment group, and point it to the appropriate SLD files. Now you'll be able to select the style profile when loading new experiments and making predictions.
