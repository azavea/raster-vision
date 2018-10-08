API
===

.. module:: rastervision

This part of the documentation lists the full API reference of public
classes and functions.

Config Builders
---------------

ExperimentConfigBuilders are created by calling

.. code::

   rv.ExperimentConfig.builder()


ExperimentConfigBuilder
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rastervision.experiment.ExperimentConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:

TaskConfig
^^^^^^^^^^

TaskConfigBuilders are created by calling

.. code::

   rv.TaskConfig.builder(<TASK_TYPE>)

Where ``<TASK_TYPE>`` is one of the following:

rv.CHIP_CLASSIFICATION
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.task.ChipClassificationConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:

BackendConfig
^^^^^^^^^^^^^

TKTK

SceneConfig
^^^^^^^^^^^

ExperimentConfigBuilders are created by calling

.. code::

   rv.SceneConfig.builder()

.. autoclass:: rastervision.data.SceneConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:


RasterSourceConfig
^^^^^^^^^^^^^^^^^^

ExperimentConfigBuilders are created by calling

.. code::

   rv.SceneConfig.builder(<SOURCE_TYPE>)


Where ``<TASK_TYPE>`` is one of the following:

rv.GEOTIFF_SOURCE
~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.GeoTiffSourceConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:

rv.IMAGE_SOURCE
~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.ImageSourceConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
