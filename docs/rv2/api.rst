.. _rv2_api:

API Reference
========================

rastervision.pipeline
------------------------

.. autoclass:: rastervision.pipeline.pipeline_config.PipelineConfig

rastervision.core
-------------------

StatsAnalyzerConfig
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.analyzer.StatsAnalyzerConfig

ClassConfig
~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.ClassConfig

DatasetConfig
~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.DatasetConfig

SceneConfig
~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.SceneConfig

ChipClassificationLabelSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_source.ChipClassificationLabelSourceConfig

SemanticSegmentationLabelSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_source.SemanticSegmentationLabelSourceConfig

ObjectDetectionLabelSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_source.ObjectDetectionLabelSourceConfig

ChipClassificationGeoJSONStoreConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_store.ChipClassificationGeoJSONStoreConfig

PolygonVectorOutputConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_store.PolygonVectorOutputConfig

BuildingVectorOutputConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_store.BuildingVectorOutputConfig

SemanticSegmentationLabelStoreConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_store.SemanticSegmentationLabelStoreConfig

ObjectDetectionGeoJSONStoreConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_store.ObjectDetectionGeoJSONStoreConfig

RasterioSourceConfig
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.raster_source.RasterioSourceConfig

RasterizedSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.raster_source.RasterizedSourceConfig

StatsTransformerConfig
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.raster_transformer.StatsTransformerConfig

VectorSourceConfig
~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.vector_source.VectorSourceConfig

GeoJSONVectorSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.vector_source.GeoJSONVectorSourceConfig

ChipClassificationEvaluatorConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.evaluation.ChipClassificationEvaluatorConfig

SemanticSegmentationEvaluatorConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.evaluation.SemanticSegmentationEvaluatorConfig

ObjectDetectionEvaluatorConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.evaluation.ObjectDetectionEvaluatorConfig

ChipClassificationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.rv_pipeline.ChipClassificationConfig

SemanticSegmentationWindowMethod
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.rv_pipeline.SemanticSegmentationWindowMethod

SemanticSegmentationChipOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.rv_pipeline.SemanticSegmentationChipOptions

SemanticSegmentationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.rv_pipeline.SemanticSegmentationConfig

ObjectDetectionWindowMethod
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.rv_pipeline.ObjectDetectionWindowMethod

ObjectDetectionChipOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.rv_pipeline.ObjectDetectionChipOptions

ObjectDetectionPredictOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.rv_pipeline.ObjectDetectionPredictOptions

ObjectDetectionConfig
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.rv_pipeline.ObjectDetectionConfig

rastervision.pytorch_backend
-------------------------------

PyTorchChipClassificationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_backend.PyTorchChipClassificationConfig

PyTorchSemanticSegmentationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_backend.PyTorchSemanticSegmentationConfig

PyTorchObjectDetectionConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_backend.PyTorchObjectDetectionConfig

rastervision.pytorch_learner
-------------------------------

Backbone
~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.Backbone
    :members:
    :undoc-members:

SolverConfig
~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.SolverConfig

ClassificationDataFormat
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ClassificationDataFormat
    :members:
    :undoc-members:

ClassificationDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ClassificationDataConfig

ClassificationModelConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ClassificationModelConfig

ClassificationLearnerConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ClassificationLearnerConfig

SemanticSegmentationDataFormat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.SemanticSegmentationDataFormat
    :members:
    :undoc-members:

SemanticSegmentationDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.SemanticSegmentationDataConfig

SemanticSegmentationModelConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.SemanticSegmentationModelConfig

SemanticSegmentationLearnerConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.SemanticSegmentationLearnerConfig

ObjectDetectionDataFormat
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ObjectDetectionDataFormat
    :members:
    :undoc-members:

ObjectDetectionDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ObjectDetectionDataConfig

ObjectDetectionModelConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ObjectDetectionModelConfig

ObjectDetectionLearnerConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ObjectDetectionLearnerConfig
