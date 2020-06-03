.. _rv2_api:

API Reference
========================

rastervision2.pipeline
------------------------

.. autoclass:: rastervision2.pipeline.pipeline_config.PipelineConfig

rastervision2.core
-------------------

StatsAnalyzerConfig
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.analyzer.StatsAnalyzerConfig

ClassConfig
~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.ClassConfig

DatasetConfig
~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.DatasetConfig

SceneConfig
~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.SceneConfig

ChipClassificationLabelSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.label_source.ChipClassificationLabelSourceConfig

SemanticSegmentationLabelSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.label_source.SemanticSegmentationLabelSourceConfig

ObjectDetectionLabelSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.label_source.ObjectDetectionLabelSourceConfig

ChipClassificationGeoJSONStoreConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.label_store.ChipClassificationGeoJSONStoreConfig

PolygonVectorOutputConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.label_store.PolygonVectorOutputConfig

BuildingVectorOutputConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.label_store.BuildingVectorOutputConfig

SemanticSegmentationLabelStoreConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.label_store.SemanticSegmentationLabelStoreConfig

ObjectDetectionGeoJSONStoreConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.label_store.ObjectDetectionGeoJSONStoreConfig

RasterioSourceConfig
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.raster_source.RasterioSourceConfig

RasterizedSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.raster_source.RasterizedSourceConfig

StatsTransformerConfig
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.raster_transformer.StatsTransformerConfig

VectorSourceConfig
~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.vector_source.VectorSourceConfig

GeoJSONVectorSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.vector_source.GeoJSONVectorSourceConfig

ChipClassificationEvaluatorConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.evaluation.ChipClassificationEvaluatorConfig

SemanticSegmentationEvaluatorConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.evaluation.SemanticSegmentationEvaluatorConfig

ObjectDetectionEvaluatorConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.evaluation.ObjectDetectionEvaluatorConfig

ChipClassificationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.rv_pipeline.ChipClassificationConfig

SemanticSegmentationWindowMethod
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.rv_pipeline.SemanticSegmentationWindowMethod

SemanticSegmentationChipOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.rv_pipeline.SemanticSegmentationChipOptions

SemanticSegmentationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.rv_pipeline.SemanticSegmentationConfig

ObjectDetectionWindowMethod
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.rv_pipeline.ObjectDetectionWindowMethod

ObjectDetectionChipOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.rv_pipeline.ObjectDetectionChipOptions

ObjectDetectionPredictOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.rv_pipeline.ObjectDetectionPredictOptions

ObjectDetectionConfig
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.rv_pipeline.ObjectDetectionConfig

rastervision2.pytorch_backend
-------------------------------

PyTorchChipClassificationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_backend.PyTorchChipClassificationConfig

PyTorchSemanticSegmentationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_backend.PyTorchSemanticSegmentationConfig

PyTorchObjectDetectionConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_backend.PyTorchObjectDetectionConfig

rastervision2.pytorch_learner
-------------------------------

Backbone
~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.Backbone
    :members:
    :undoc-members:

SolverConfig
~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.SolverConfig

ClassificationDataFormat
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.ClassificationDataFormat
    :members:
    :undoc-members:

ClassificationDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.ClassificationDataConfig

ClassificationModelConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.ClassificationModelConfig

ClassificationLearnerConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.ClassificationLearnerConfig

SemanticSegmentationDataFormat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.SemanticSegmentationDataFormat
    :members:
    :undoc-members:

SemanticSegmentationDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.SemanticSegmentationDataConfig

SemanticSegmentationModelConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.SemanticSegmentationModelConfig

SemanticSegmentationLearnerConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.SemanticSegmentationLearnerConfig

ObjectDetectionDataFormat
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.ObjectDetectionDataFormat
    :members:
    :undoc-members:

ObjectDetectionDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.ObjectDetectionDataConfig

ObjectDetectionModelConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.ObjectDetectionModelConfig

ObjectDetectionLearnerConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.ObjectDetectionLearnerConfig
