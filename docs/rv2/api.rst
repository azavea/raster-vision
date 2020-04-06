.. _rv2_api:

API Reference
========================

This currently just contains references for ``Config`` classes related to chip classification and semantic segmentation.

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

.. autoclass:: rastervision2.core.data.label_store.ChipClassificationGeoJSONStoreConfig

.. autoclass:: rastervision2.core.data.label_store.PolygonVectorOutputConfig

.. autoclass:: rastervision2.core.data.label_store.BuildingVectorOutputConfig

.. autoclass:: rastervision2.core.data.label_store.SemanticSegmentationLabelStoreConfig

.. autoclass:: rastervision2.core.data.raster_source.RasterioSourceConfig

.. autoclass:: rastervision2.core.data.raster_source.RasterizedSourceConfig

.. autoclass:: rastervision2.core.data.raster_transformer.StatsTransformerConfig

.. autoclass:: rastervision2.core.data.vector_source.VectorSourceConfig

.. autoclass:: rastervision2.core.data.vector_source.GeoJSONVectorSourceConfig

evaluation
~~~~~~~~~~~

.. autoclass:: rastervision2.core.evaluation.ChipClassificationEvaluatorConfig

.. autoclass:: rastervision2.core.evaluation.SemanticSegmentationEvaluatorConfig

rv_pipeline
~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.rv_pipeline.ChipClassificationConfig

.. autoclass:: rastervision2.core.rv_pipeline.SemanticSegmentationChipOptions

.. autoclass:: rastervision2.core.rv_pipeline.SemanticSegmentationConfig

rastervision2.pytorch_backend
-------------------------------

.. autoclass:: rastervision2.pytorch_backend.PyTorchChipClassificationConfig

.. autoclass:: rastervision2.pytorch_backend.PyTorchSemanticSegmentationConfig

rastervision2.pytorch_learner
-------------------------------

.. autoclass:: rastervision2.pytorch_learner.SolverConfig

Classification
~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.ClassificationDataConfig

.. autoclass:: rastervision2.pytorch_learner.ClassificationModelConfig

.. autoclass:: rastervision2.pytorch_learner.ClassificationLearnerConfig

Semantic Segmentation
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.pytorch_learner.SemanticSegmentationDataConfig

.. autoclass:: rastervision2.pytorch_learner.SemanticSegmentationModelConfig

.. autoclass:: rastervision2.pytorch_learner.SemanticSegmentationLearnerConfig
