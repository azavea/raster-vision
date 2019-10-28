API Reference
=============

.. module:: rastervision

This API documentation is not exhaustive, but covers most of the public API that is important to typical Raster Vision usage.

ExperimentConfigBuilder
-----------------------

An ExperimentConfigBuilder is created by calling

.. code::

   rv.ExperimentConfig.builder()

.. autoclass:: rastervision.experiment.ExperimentConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

DatasetConfigBuilder
--------------------

A DatasetConfigBuilder is created by calling

.. code::

   rv.DatasetConfig.builder()

.. autoclass:: rastervision.data.DatasetConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

.. _task api reference:

TaskConfigBuilder
-----------------

TaskConfigBuilders are created by calling

.. code::

   rv.TaskConfig.builder(TASK_TYPE)

Where ``TASK_TYPE`` is one of the following:

rv.CHIP_CLASSIFICATION
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.task.ChipClassificationConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

rv.OBJECT_DETECTION
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.task.ObjectDetectionConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

rv.SEMANTIC_SEGMENTATION
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.task.SemanticSegmentationConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

.. _backend api reference:

BackendConfig
-------------

There are backends based on PyTorch and Tensorflow. Remember to use the appropriate Docker image depending on the backend. Note that the Tensorflow backends are being sunsetted.
BackendConfigBuilders are created by calling

.. code::

   rv.BackendConfig.builder(BACKEND_TYPE)

Where ``BACKEND_TYPE`` is one of the following:

rv.PYTORCH_SEMANTIC_SEGMENTATION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.backend.pytorch_semantic_segmentation_config.PyTorchSemanticSegmentationConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

rv.PYTORCH_CHIP_CLASSIFICATION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.backend.pytorch_chip_classification_config.PyTorchChipClassificationConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

rv.PYTORCH_OBJECT_DETECTION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.backend.pytorch_object_detection_config.PyTorchObjectDetectionConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate


rv.KERAS_CLASSIFICATION
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.backend.keras_classification_config.KerasClassificationConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

rv.TF_OBJECT_DETECTION
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.backend.tf_object_detection_config.TFObjectDetectionConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

rv.TF_DEEPLAB
~~~~~~~~~~~~~

.. autoclass:: rastervision.backend.tf_deeplab_config.TFDeeplabConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

SceneConfig
-----------

SceneConfigBuilders are created by calling

.. code::

   rv.SceneConfig.builder()

.. autoclass:: rastervision.data.SceneConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

.. _raster source api reference:

RasterSourceConfig
------------------

RasterSourceConfigBuilders are created by calling

.. code::

   rv.RasterSourceConfig.builder(SOURCE_TYPE)


Where ``SOURCE_TYPE`` is one of the following:

rv.RASTERIO_SOURCE
~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.RasterioSourceConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

rv.RASTERIZED_SOURCE
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.RasterizedSourceConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

.. _label source api reference:

LabelSourceConfig
-----------------

LabelSourceConfigBuilders are created by calling

.. code::

   rv.LabelSourceConfig.builder(SOURCE_TYPE)


Where ``SOURCE_TYPE`` is one of the following:

rv.CHIP_CLASSIFICATION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.ChipClassificationLabelSourceConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

rv.OBJECT_DETECTION
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.ObjectDetectionLabelSourceConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

rv.SEMANTIC_SEGMENTATION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.SemanticSegmentationLabelSourceConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

.. _vector source api reference:

VectorSourceConfig
--------------------

VectorSourceConfigBuilders are created by calling

.. code::

  rv.VectorSourceConfig.builder(SOURCE_TYPE)


Where ``SOURCE_TYPE`` is one of the following:

rv.GEOJSON_SOURCE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.GeoJSONVectorSourceConfigBuilder
  :members:
  :undoc-members:
  :inherited-members:
  :exclude-members: from_proto, validate

rv.VECTOR_TILE_SOURCE
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.VectorTileVectorSourceConfigBuilder
  :members:
  :undoc-members:
  :inherited-members:
  :exclude-members: from_proto, validate

.. _label store api reference:

LabelStoreConfig
-----------------

LabelStoreConfigBuilders are created by calling

.. code::

   rv.LabelStoreConfig.builder(STORE_TYPE)


Where ``STORE_TYPE`` is one of the following:

rv.CHIP_CLASSIFICATION_GEOJSON
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.ChipClassificationGeoJSONStoreConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

For ``rv.OBJECT_DETECTION``:


rv.OBJECT_DETECTION_GEOJSON
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.ObjectDetectionGeoJSONStoreConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

rv.SEMANTIC_SEGMENTATION_RASTER
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.SemanticSegmentationRasterStoreConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

.. _raster transformer api reference:

RasterTransformerConfig
-----------------------

RasterTransformerConfigBuilders are created by calling

.. code::

   rv.RasterTransformerConfig.builder(TRANSFORMER_TYPE)

Where ``TRANSFORMER_TYPE`` is one of the following:

rv.STATS_TRANSFORMER
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.data.StatsTransformerConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

.. _augmentor api reference:

AugmentorConfig
---------------

AugmentorConfigBuilders are created by calling

.. code::

   rv.AugmentorConfig.builder(AUGMENTOR_TYPE)

Where ``AUGMENTOR_TYPE`` is one of the following:

rv.NODATA_AUGMENTOR
~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.augmentor.NodataAugmentorConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

.. _analyzer api reference:

AnalyzerConfig
--------------

AnalyzerConfigBuilders are created by calling

.. code::

   rv.AnalyzerConfig.builder(ANALYZER_TYPE)

Where ``ANALYZER_TYPE`` is one of the following:

rv.STATS_ANALYZER
~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.analyzer.StatsAnalyzerConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

.. _evaluator api reference:

EvaluatorConfig
---------------

EvaluatorConfigBuilders are created by calling

.. code::

   rv.EvaluatorConfig.builder(Evaluator_TYPE)

Where ``Evaluator_TYPE`` is one of the following:

rv.CHIP_CLASSIFICATION_EVALUATOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.evaluation.ChipClassificationEvaluatorConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

rv.OBJECT_DETECTION_EVALUATOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.evaluation.ObjectDetectionEvaluatorConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

rv.SEMANTIC_SEGMENTATION_EVALUATOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.evaluation.SemanticSegmentationEvaluatorConfigBuilder
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: from_proto, validate

.. _aux command api:

Aux Commands
------------

.. autoclass:: rastervision.command.aux.CogifyCommand

.. _aux command options api:

Aux Command Options
-------------------

.. autoclass:: rastervision.command.aux_command.AuxCommandOptions

   .. automethod:: __init__


.. _predictor api:

Predictor
---------

.. autoclass:: rastervision.Predictor
   :members:
   :undoc-members:
   :inherited-members:

   .. automethod:: __init__

.. _plugin registry api:

Plugin Registry
---------------

.. autoclass:: rastervision.plugin.PluginRegistry
   :members:
   :undoc-members:
   :inherited-members:
   :exclude-members: add_plugins_from_proto, get_instance, to_proto
