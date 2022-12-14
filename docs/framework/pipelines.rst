{% set tiny_spacenet = '`tiny_spacenet.py <' ~ repo_examples ~ '/tiny_spacenet.py>`__' %}
{% set isprs_potsdam = '`isprs_potsdam.py <' ~ repo_examples ~ '/semantic_segmentation/isprs_potsdam.py>`__' %}

.. _rv pipelines:

Pipelines and Commands
======================

In addition to providing abstract :ref:`pipeline <rv pipelines>` functionality, Raster Vision provides a set of concrete pipelines for deep learning on remote sensing imagery including :class:`~rastervision.core.rv_pipeline.chip_classification.ChipClassification`, :class:`~rastervision.core.rv_pipeline.semantic_segmentation.SemanticSegmentation`, and :class:`~rastervision.core.rv_pipeline.object_detection.ObjectDetection`. These pipelines all derive from :class:`~rastervision.core.rv_pipeline.rv_pipeline.RVPipeline`, and are provided by the :mod:`rastervision.core` package. It's possible to customize these pipelines as well as create new ones from scratch, which is discussed in :ref:`customizing rv`.

.. image:: /img/cv-tasks.png
    :width: 75%
    :align: center

Chip Classification
-------------------

In chip classification, the goal is to divide the scene up into a grid of cells and classify each cell. This task is good for getting a rough idea of where certain objects are located, or where indiscrete "stuff" (such as grass) is located. It requires relatively low labeling effort, but also produces spatially coarse predictions. In our experience, this task trains the fastest, and is easiest to configure to get "decent" results.

Object Detection
----------------

In object detection, the goal is to predict a bounding box and a class around each object of interest. This task requires higher labeling effort than chip classification, but has the ability to localize and individuate objects. Object detection models require more time to train and also struggle with objects that are very close together. In theory, it is straightforward to use object detection for counting objects.

Semantic Segmentation
---------------------

In semantic segmentation, the goal is to predict the class of each pixel in a scene. This task requires the highest labeling effort, but also provides the most spatially precise predictions. Like object detection, these models take longer to train than chip classification models.

.. _configuring rvpipelines:

Configuring RVPipelines
-----------------------

Each (subclass of) :class:`~rastervision.core.rv_pipeline.rv_pipeline.RVPipeline` is configured by returning an instance of (a subclass of) :class:`RVPipelineConfigs <rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig>` from a ``get_config()`` function in a Python module. It's also possible to return a list of :class:`RVPipelineConfigs <rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig>` from ``get_configs()``, which will be executed in parallel.

Each :class:`~rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig` object specifies the details about how the commands within the pipeline will execute (ie. which files, what methods, what hyperparameters, etc.). In contrast, the :ref:`pipeline runner <runners>`, which actually executes the commands in the pipeline, is specified as an argument to the :ref:`cli`. The following diagram shows the hierarchy of the high level components comprising an :class:`~rastervision.core.rv_pipeline.rv_pipeline.RVPipeline`:

.. image:: /img/rvpipeline-diagram.png
    :align: center

In the {{ tiny_spacenet }} example, the :class:`~rastervision.core.rv_pipeline.semantic_segmentation_config.SemanticSegmentationConfig` is the last thing constructed and returned from the ``get_config`` function.

.. literalinclude:: /../rastervision_pytorch_backend/rastervision/pytorch_backend/examples/tiny_spacenet.py
    :language: python
    :lines: 48-53
    :dedent:

.. seealso:: The :class:`~rastervision.core.rv_pipeline.chip_classification_config.ChipClassificationConfig`, :class:`~rastervision.core.rv_pipeline.semantic_segmentation_config.SemanticSegmentationConfig`, and :class:`~rastervision.core.rv_pipeline.object_detection_config.ObjectDetectionConfig` API docs have more information on configuring pipelines.

Commands
--------

The :class:`RVPipelines <rastervision.core.rv_pipeline.rv_pipeline.RVPipeline>` provide a sequence of commands, which are described below.

.. image:: /img/rv-pipeline-overview.png
    :align: center
    :class: only-light

.. image:: /img/rv-pipeline-overview.png
    :align: center
    :class: only-dark

ANALYZE
^^^^^^^

The ANALYZE command is used to analyze scenes that are part of an experiment and produce some output that can be consumed by later commands. Geospatial raster sources such as GeoTIFFs often contain 16- and 32-bit pixel color values, but many deep learning libraries expect 8-bit values. In order to perform this transformation, we need to know the distribution of pixel values. So one usage of the ANALYZE command is to compute statistics of the raster sources and save them to a JSON file which is later used by the StatsTransformer (one of the available :class:`RasterTransformers <rastervision.core.data.raster_transformer.raster_transformer.RasterTransformer>`) to do the conversion.

.. _chip command:

CHIP
^^^^

Scenes comprise large geospatial raster sources (e.g. GeoTIFFs) and geospatial label sources (e.g. GeoJSONs), but models can only consume small images (i.e. chips) and labels in pixel based-coordinates. In addition, each :ref:`backend` has its own dataset format. The CHIP command solves this problem by converting scenes into training chips and into a format the backend can use for training.

TRAIN
^^^^^

The TRAIN command is used to train a model using the dataset generated by the CHIP command. The command uses the :ref:`backend` to run a training loop that saves the model and other artifacts each epoch. If the training command is interrupted, it will resume at the last epoch when restarted.

.. _predict command:

PREDICT
^^^^^^^

The PREDICT command makes predictions for a set of scenes using a model produced by the TRAIN command. To do this, a sliding window is used to feed small images into the model, and the predictions are transformed from image-centric, pixel-based coordinates into scene-centric, map-based coordinates.

EVAL
^^^^

The EVAL command evaluates the quality of models by comparing the predictions generated by the PREDICT command to ground truth labels. A variety of metrics including F1, precision, and recall are computed for each class (as well as overall) and are written to a JSON file.

.. _bundle command:

BUNDLE
^^^^^^

The BUNDLE command generates a model bundle from the output of the previous commands which contains a model file plus associated configuration data. A model bundle can be used to make predictions on new imagery using the :ref:`predict cli command` command.


Pipeline components
-------------------

Below we describe some components of the pipeline that you might directly or indirectly configure.

A lot of these will be familiar from :doc:`../usage/basics`, but note that when configuring a pipeline, instead of dealing with the classes directly, you will instead be configuring their ``Config`` counterparts.

The following table shows the corresponding ``Configs`` for various commonly used classes.

.. currentmodule:: rastervision.core

.. list-table::
   :header-rows: 1

   * -  Class
     -  Config
     -  Notes
   * -  :class:`~data.scene.Scene`
     -  :class:`~data.scene_config.SceneConfig`
     -  :ref:`notes <scene>`
   * -  :class:`~data.raster_source.raster_source.RasterSource`


        - :class:`~data.raster_source.rasterio_source.RasterioSource`
        - :class:`~data.raster_source.multi_raster_source.MultiRasterSource`
        - :class:`~data.raster_source.rasterized_source.RasterizedSource`

     -  :class:`~data.raster_source.raster_source_config.RasterSourceConfig`

        - :class:`~data.raster_source.rasterio_source_config.RasterioSourceConfig`
        - :class:`~data.raster_source.multi_raster_source_config.MultiRasterSourceConfig`
        - :class:`~data.raster_source.rasterized_source_config.RasterizedSourceConfig`

     -  
   * -  :class:`~data.raster_transformer.raster_transformer.RasterTransformer`

        - :class:`~data.raster_transformer.cast_transformer.CastTransformer`
        - :class:`~data.raster_transformer.min_max_transformer.MinMaxTransformer`
        - :class:`~data.raster_transformer.nan_transformer.NanTransformer`
        - :class:`~data.raster_transformer.reclass_transformer.ReclassTransformer`
        - :class:`~data.raster_transformer.rgb_class_transformer.RGBClassTransformer`
        - :class:`~data.raster_transformer.stats_transformer.StatsTransformer`

     -  :class:`~data.raster_transformer.raster_transformer_config.RasterTransformerConfig`

        - :class:`~data.raster_transformer.cast_transformer_config.CastTransformerConfig`
        - :class:`~data.raster_transformer.min_max_transformer_config.MinMaxTransformerConfig`
        - :class:`~data.raster_transformer.nan_transformer_config.NanTransformerConfig`
        - :class:`~data.raster_transformer.reclass_transformer_config.ReclassTransformerConfig`
        - :class:`~data.raster_transformer.rgb_class_transformer_config.RGBClassTransformerConfig`
        - :class:`~data.raster_transformer.stats_transformer_config.StatsTransformerConfig`

     -  
   * -  :class:`~data.vector_source.vector_source.VectorSource`

        - :class:`~data.vector_source.geojson_vector_source.GeoJSONVectorSource`

     -  :class:`~data.vector_source.vector_source_config.VectorSourceConfig`

        - :class:`~data.vector_source.geojson_vector_source_config.GeoJSONVectorSourceConfig`

     -  
   * -  :class:`~data.vector_transformer.vector_transformer.VectorTransformer`

        - :class:`~data.vector_transformer.buffer_transformer.BufferTransformer`
        - :class:`~data.vector_transformer.class_inference_transformer.ClassInferenceTransformer`
        - :class:`~data.vector_transformer.shift_transformer.ShiftTransformer`

     -  :class:`~data.vector_transformer.vector_transformer_config.VectorTransformerConfig`

        - :class:`~data.vector_transformer.buffer_transformer_config.BufferTransformerConfig`
        - :class:`~data.vector_transformer.class_inference_transformer_config.ClassInferenceTransformerConfig`
        - :class:`~data.vector_transformer.shift_transformer_config.ShiftTransformerConfig`

     -  
   * -  :class:`~data.label_source.label_source.LabelSource`

        - :class:`~data.label_source.chip_classification_label_source.ChipClassificationLabelSource`
        - :class:`~data.label_source.semantic_segmentation_label_source.SemanticSegmentationLabelSource`
        - :class:`~data.label_source.object_detection_label_source.ObjectDetectionLabelSource`

     -  :class:`~data.label_source.label_source_config.LabelSourceConfig`

        - :class:`~data.label_source.chip_classification_label_source_config.ChipClassificationLabelSourceConfig`
        - :class:`~data.label_source.semantic_segmentation_label_source_config.SemanticSegmentationLabelSourceConfig`
        - :class:`~data.label_source.object_detection_label_source_config.ObjectDetectionLabelSourceConfig`

     -  
   * -  :class:`~data.label_store.label_store.LabelStore`

        - :class:`~data.label_store.chip_classification_geojson_store.ChipClassificationGeoJSONStore`
        - :class:`~data.label_store.object_detection_geojson_store.ObjectDetectionGeoJSONStore`
        - :class:`~data.label_store.semantic_segmentation_label_store.SemanticSegmentationLabelStore`

     -  :class:`~data.label_store.label_store_config.LabelStoreConfig`

        - :class:`~data.label_store.chip_classification_geojson_store_config.ChipClassificationGeoJSONStoreConfig`
        - :class:`~data.label_store.object_detection_geojson_store_config.ObjectDetectionGeoJSONStoreConfig`
        - :class:`~data.label_store.semantic_segmentation_label_store_config.SemanticSegmentationLabelStoreConfig`

     -  :ref:`notes <label store>`
   * -  :class:`~analyzer.analyzer.Analyzer`

        - :class:`~analyzer.stats_analyzer.StatsAnalyzer`

     -  :class:`~analyzer.analyzer_config.AnalyzerConfig`

        - :class:`~analyzer.stats_analyzer_config.StatsAnalyzerConfig`

     -  :ref:`notes <analyzer>`
   * -  :class:`~evaluation.evaluator.Evaluator`

        - :class:`~evaluation.chip_classification_evaluator.ChipClassificationEvaluator`
        - :class:`~evaluation.semantic_segmentation_evaluator.SemanticSegmentationEvaluator`
        - :class:`~evaluation.object_detection_evaluator.ObjectDetectionEvaluator`

     -  :class:`~evaluation.evaluator_config.EvaluatorConfig`

        - :class:`~evaluation.chip_classification_evaluator_config.ChipClassificationEvaluatorConfig`
        - :class:`~evaluation.semantic_segmentation_evaluator_config.SemanticSegmentationEvaluatorConfig`
        - :class:`~evaluation.object_detection_evaluator_config.ObjectDetectionEvaluatorConfig`

     -  :ref:`notes <evaluator>`

.. _backend:

Backend
^^^^^^^

.. currentmodule:: rastervision.core

:class:`RVPipelines <rv_pipeline.rv_pipeline.RVPipeline>` use a "backend" abstraction inspired by `Keras <https://keras.io/backend/>`_, which makes it easier to customize the code for building and training models (including using Raster Vision with arbitrary deep learning libraries).
Each backend is a subclass of :class:`~backend.backend.Backend` and has methods for saving training chips, training models, and making predictions, and is configured with a :class:`~Backend <backend.backend_config.BackendConfig>`.

The :mod:`rastervision.pytorch_backend` plugin provides backends that are thin wrappers around the :mod:`rastervision.pytorch_learner` package, which does most of the heavy lifting of building and training models using `torch <https://pytorch.org/docs/stable/>`_ and `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`_. (Note that :mod:`rastervision.pytorch_learner` is decoupled from :mod:`rastervision.pytorch_backend` so that it can be used in conjunction with :mod:`rastervision.pipeline` to write arbitrary computer vision pipelines that have nothing to do with remote sensing.)

Here are the PyTorch backends:

.. currentmodule:: rastervision.pytorch_backend

* The :class:`~pytorch_chip_classification.PyTorchChipClassification` backend trains classification models from `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`_.
* The :class:`~pytorch_object_detection.PyTorchObjectDetection` backend trains the Faster-RCNN model in `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`_.
* The :class:`~pytorch_semantic_segmentation.PyTorchSemanticSegmentation` backend trains the DeepLabV3 model in `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`_.

In our {{ tiny_spacenet }} example, we configured the PyTorch semantic segmentation backend using:

.. literalinclude:: /../rastervision_pytorch_backend/rastervision/pytorch_backend/examples/tiny_spacenet.py
    :language: python
    :lines: 32-46
    :dedent:

.. seealso:: :mod:`rastervision.pytorch_backend` and :mod:`rastervision.pytorch_learner` API docs for more information on configuring backends.

.. _dataset:

DatasetConfig
^^^^^^^^^^^^^

.. currentmodule:: rastervision.core.data

A :class:`~dataset_config.DatasetConfig` defines the `training, validation, and test splits <https://en.wikipedia.org/wiki/Training,_test,_and_validation_sets>`_ needed to train and evaluate a model. Each dataset split is a list of `SceneConfigs <SceneConfig>`_.

In our {{ tiny_spacenet }} example, we configured the dataset with single scenes, though more often in real use cases you would use multiple scenes per split:

.. literalinclude:: /../rastervision_pytorch_backend/rastervision/pytorch_backend/examples/tiny_spacenet.py
    :language: python
    :lines: 23-30
    :dedent:

.. _scene:

Scene
^^^^^

.. currentmodule:: rastervision.core.data

A :ref:`usage_scene` is configured via a :class:`~scene_config.SceneConfig` which is composed of the following elements:

* *Imagery*: a :class:`~raster_source.raster_source_config.RasterSourceConfig` represents a large scene image, which can be made up of multiple sub-images or a single file.
* *Ground truth labels*: a :class:`~label_source.label_source_config.LabelSourceConfig` represents ground-truth task-specific labels.
* *Predicted labels* (Optional): a :class:`~label_store.label_store_config.LabelStoreConfig` specifies how to store and retrieve the predictions from a scene.
* *AOIs* (Optional): An optional list of areas of interest that describes which sections of the scene imagery are exhaustively labeled. It is important to only create training chips from parts of the scenes that have been exhaustively labeled -- in other words, that have no missing labels.

In our {{ tiny_spacenet }} example, we configured the one training scene with a GeoTIFF URI and a GeoJSON URI.

.. literalinclude:: /../rastervision_pytorch_backend/rastervision/pytorch_backend/examples/tiny_spacenet.py
    :language: python
    :pyobject: make_scene
    :dedent:

.. _label store:

LabelStore
^^^^^^^^^^

.. currentmodule:: rastervision.core.data.label_store

A :ref:`usage_label_store` is configured via a :class:`~label_store_config.LabelStoreConfig`.

In the {{ tiny_spacenet }} example, there is no explicit :class:`~label_store.LabelStore` configured on the validation scene, because it can be inferred from the type of :class:`~rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig` it is part of.
In the {{ isprs_potsdam }} example, the following code is used to explicitly create a :class:`~label_store_config.LabelStoreConfig` that, in turn, will be used to create a :class:`~label_store.LabelStore` that writes out the predictions in "RGB" format, where the color of each pixel represents the class, and predictions of class 0 (ie. car) are also written out as polygons.

.. code-block:: python

    label_store = SemanticSegmentationLabelStoreConfig(
        rgb=True, vector_output=[PolygonVectorOutputConfig(class_id=0)])

    scene = SceneConfig(
        id=id,
        raster_source=raster_source,
        label_source=label_source,
        label_store=label_store)

.. _analyzer:

Analyzer
^^^^^^^^

.. currentmodule:: rastervision.core.analyzer

:class:`Analyzers <analyzer.Analyzer>`, configured via :class:`AnalyzerConfigs <analyzer_config.AnalyzerConfig>`, are used to gather dataset-level statistics and metrics for use in downstream processes. Typically, you won't need to explicitly configure any.

.. _evaluator:

Evaluator
^^^^^^^^^

.. currentmodule:: rastervision.core.evaluation

For each computer vision task, there is an :class:`~evaluator.Evaluator` (configured via the corresponding :class:`~evaluator_config.EvaluatorConfig`) that computes metrics for a trained model. It does this by measuring the discrepancy between ground truth and predicted labels for a set of validation scenes. Typically, you won't need to explicitly configure any.
