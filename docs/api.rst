.. _api:

Configuration API Reference
============================

This contains the API used for configuring various components of Raster Vision pipelines. This serves as the lower-level companion to the discussion of :ref:`rv pipelines`.

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

.. _api DatasetConfig:

DatasetConfig
~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.DatasetConfig

.. _api SceneConfig:

SceneConfig
~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.SceneConfig

.. _api ChipClassificationLabelSourceConfig:

ChipClassificationLabelSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_source.ChipClassificationLabelSourceConfig

.. _api SemanticSegmentationLabelSourceConfig:

SemanticSegmentationLabelSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_source.SemanticSegmentationLabelSourceConfig

.. _api ObjectDetectionLabelSourceConfig:

ObjectDetectionLabelSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_source.ObjectDetectionLabelSourceConfig

.. _api ChipClassificationGeoJSONStoreConfig:

ChipClassificationGeoJSONStoreConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_store.ChipClassificationGeoJSONStoreConfig

PolygonVectorOutputConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_store.PolygonVectorOutputConfig

BuildingVectorOutputConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_store.BuildingVectorOutputConfig

.. _api SemanticSegmentationLabelStoreConfig:

SemanticSegmentationLabelStoreConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_store.SemanticSegmentationLabelStoreConfig

.. _api ObjectDetectionGeoJSONStoreConfig:

ObjectDetectionGeoJSONStoreConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.label_store.ObjectDetectionGeoJSONStoreConfig

.. _api RasterioSourceConfig:

RasterioSourceConfig
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.raster_source.RasterioSourceConfig

RasterizerConfig
~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.raster_source.RasterizerConfig

.. _api RasterizedSourceConfig:

RasterizedSourceConfig
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.raster_source.RasterizedSourceConfig

.. _api StatsTransformerConfig:

StatsTransformerConfig
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.raster_transformer.StatsTransformerConfig

VectorSourceConfig
~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.data.vector_source.VectorSourceConfig

.. _api GeoJSONVectorSourceConfig:

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

.. _api ChipClassificationConfig:

ChipClassificationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.rv_pipeline.ChipClassificationConfig

SemanticSegmentationWindowMethod
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.rv_pipeline.SemanticSegmentationWindowMethod

SemanticSegmentationChipOptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.rv_pipeline.SemanticSegmentationChipOptions

.. _api SemanticSegmentationConfig:

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

.. _api ObjectDetectionConfig:

ObjectDetectionConfig
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.core.rv_pipeline.ObjectDetectionConfig

.. _api rastervision.pytorch_backend:

rastervision.pytorch_backend
-------------------------------

.. _api PyTorchChipClassificationConfig:

PyTorchChipClassificationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_backend.PyTorchChipClassificationConfig

.. _api PyTorchSemanticSegmentationConfig:

PyTorchSemanticSegmentationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_backend.PyTorchSemanticSegmentationConfig

.. _api PyTorchObjectDetectionConfig:

PyTorchObjectDetectionConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_backend.PyTorchObjectDetectionConfig

.. _api rastervision.pytorch_learner:

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

ExternalModuleConfig
~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ExternalModuleConfig

PlotOptions
~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.PlotOptions

ClassificationDataFormat
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ClassificationDataFormat
    :members:
    :undoc-members:

ClassificationDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ClassificationDataConfig

ClassificationImageDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ClassificationImageDataConfig

ClassificationGeoDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ClassificationGeoDataConfig

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

SemanticSegmentationImageDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.SemanticSegmentationImageDataConfig

SemanticSegmentationGeoDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.SemanticSegmentationGeoDataConfig

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

ObjectDetectionImageDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ObjectDetectionImageDataConfig

ObjectDetectionGeoDataConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ObjectDetectionGeoDataConfig

ObjectDetectionModelConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ObjectDetectionModelConfig

ObjectDetectionLearnerConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision.pytorch_learner.ObjectDetectionLearnerConfig
