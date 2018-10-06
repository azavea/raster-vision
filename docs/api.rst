API
===

.. module:: rastervision

This part of the documentation lists the full API reference of public
classes and functions.

Config Builders
---------------

ExperimentConfigBuilder
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rastervision.experiment.ExperimentConfigBuilder
   :members:

TaskConfig
^^^^^^^^^^

TaskConfigBuilders are created by calling

.. code::

   rv.TaskConfig.builder(<TASK_TYPE>)

Where ``<TASK_TYPE>`` is one of the following:

.. autoclass:: rastervision.task.TaskConfigBuilder
   :members:

.. autoclass:: rastervision.task.ChipClassificationConfigBuilder
   :members:
