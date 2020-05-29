.. _rv2_api:

API Reference
========================

rastervision2.pipeline
------------------------

.. autoclass:: rastervision2.pipeline.pipeline_config.PipelineConfig

rastervision2.core
-------------------

analyzer
~~~~~~~~~~~

.. autoclass:: rastervision2.core.analyzer.StatsAnalyzerConfig

data
~~~~~

.. autoclass:: rastervision2.core.data.ClassConfig

.. autoclass:: rastervision2.core.data.DatasetConfig

.. autoclass:: rastervision2.core.data.SceneConfig

.. autoclass:: rastervision2.core.data.label_source.ChipClassificationLabelSourceConfig

.. autoclass:: rastervision2.core.data.label_source.SemanticSegmentationLabelSourceConfig

.. autoclass:: rastervision2.core.data.label_source.ObjectDetectionLabelSourceConfig

.. autoclass:: rastervision2.core.data.label_store.ChipClassificationGeoJSONStoreConfig

.. autoclass:: rastervision2.core.data.label_store.PolygonVectorOutputConfig

.. autoclass:: rastervision2.core.data.label_store.BuildingVectorOutputConfig

.. autoclass:: rastervision2.core.data.label_store.SemanticSegmentationLabelStoreConfig

.. autoclass:: rastervision2.core.data.label_store.ObjectDetectionGeoJSONStoreConfig

.. autoclass:: rastervision2.core.data.raster_source.RasterioSourceConfig

.. autoclass:: rastervision2.core.data.raster_source.RasterizedSourceConfig

.. autoclass:: rastervision2.core.data.raster_transformer.StatsTransformerConfig

.. autoclass:: rastervision2.core.data.vector_source.VectorSourceConfig

.. autoclass:: rastervision2.core.data.vector_source.GeoJSONVectorSourceConfig

evaluation
~~~~~~~~~~~

.. autoclass:: rastervision2.core.evaluation.ChipClassificationEvaluatorConfig

.. autoclass:: rastervision2.core.evaluation.SemanticSegmentationEvaluatorConfig

.. autoclass:: rastervision2.core.evaluation.ObjectDetectionEvaluatorConfig

rv_pipeline
~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.rv_pipeline.ChipClassificationConfig

.. autoclass:: rastervision2.core.rv_pipeline.SemanticSegmentationWindowMethod

.. autoclass:: rastervision2.core.rv_pipeline.SemanticSegmentationChipOptions

.. autoclass:: rastervision2.core.rv_pipeline.SemanticSegmentationConfig

.. autoclass:: rastervision2.core.rv_pipeline.ObjectDetectionWindowMethod

.. autoclass:: rastervision2.core.rv_pipeline.ObjectDetectionChipOptions

.. autoclass:: rastervision2.core.rv_pipeline.ObjectDetectionPredictOptions

.. autoclass:: rastervision2.core.rv_pipeline.ObjectDetectionConfig

rastervision2.pytorch_backend
-------------------------------

.. autoclass:: rastervision2.pytorch_backend.PyTorchChipClassificationConfig

.. autoclass:: rastervision2.pytorch_backend.PyTorchSemanticSegmentationConfig

.. autoclass:: rastervision2.pytorch_backend.PyTorchObjectDetectionConfig

rastervision2.pytorch_learner
-------------------------------

.. autoclass:: rastervision2.pytorch_learner.Backbone
    :members:
    :undoc-members:

.. autoclass:: rastervision2.pytorch_learner.SolverConfig

Classification
~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.ClassificationDataFormat
    :members:
    :undoc-members:

.. autoclass:: rastervision2.pytorch_learner.ClassificationDataConfig

.. autoclass:: rastervision2.pytorch_learner.ClassificationModelConfig

.. autoclass:: rastervision2.pytorch_learner.ClassificationLearnerConfig

Semantic Segmentation
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.SemanticSegmentationDataFormat
    :members:
    :undoc-members:

.. autoclass:: rastervision2.pytorch_learner.SemanticSegmentationDataConfig

.. autoclass:: rastervision2.pytorch_learner.SemanticSegmentationModelConfig

.. autoclass:: rastervision2.pytorch_learner.SemanticSegmentationLearnerConfig

Object Detection
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.ObjectDetectionDataFormat
    :members:
    :undoc-members:

.. autoclass:: rastervision2.pytorch_learner.ObjectDetectionDataConfig

.. autoclass:: rastervision2.pytorch_learner.ObjectDetectionModelConfig

.. autoclass:: rastervision2.pytorch_learner.ObjectDetectionLearnerConfig
