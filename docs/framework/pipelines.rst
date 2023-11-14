{% set tiny_spacenet = '`tiny_spacenet.py <' ~ repo_examples ~ '/tiny_spacenet.py>`__' %}
{% set isprs_potsdam = '`isprs_potsdam.py <' ~ repo_examples ~ '/semantic_segmentation/isprs_potsdam.py>`__' %}

.. _rv pipelines:

Pipelines and Commands
======================

In addition to providing abstract :ref:`pipeline <rv pipelines>` functionality, Raster Vision provides a set of concrete pipelines for deep learning on remote sensing imagery including :class:`.ChipClassification`, :class:`.SemanticSegmentation`, and :class:`.ObjectDetection`. These pipelines all derive from :class:`.RVPipeline`, and are provided by the :mod:`rastervision.core` package. It's possible to customize these pipelines as well as create new ones from scratch, which is discussed in :ref:`customizing rv`.

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

Each (subclass of) :class:`.RVPipeline` is configured by returning an instance of (a subclass of) :class:`RVPipelineConfigs <rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig>` from a ``get_config()`` function in a Python module. It's also possible to return a list of :class:`RVPipelineConfigs <rastervision.core.rv_pipeline.rv_pipeline_config.RVPipelineConfig>` from ``get_configs()``, which will be executed in parallel.

Each :class:`.RVPipelineConfig` object specifies the details about how the commands within the pipeline will execute (ie. which files, what methods, what hyperparameters, etc.). In contrast, the :ref:`pipeline runner <runners>`, which actually executes the commands in the pipeline, is specified as an argument to the :ref:`cli`. The following diagram shows the hierarchy of the high level components comprising an :class:`.RVPipeline`:

.. image:: /img/rvpipeline-diagram.png
    :align: center

In the {{ tiny_spacenet }} example, the :class:`.SemanticSegmentationConfig` is the last thing constructed and returned from the ``get_config`` function.

.. literalinclude:: /../rastervision_pytorch_backend/rastervision/pytorch_backend/examples/tiny_spacenet.py
    :language: python
    :lines: 48-53
    :dedent:

.. seealso:: The :class:`.ChipClassificationConfig`, :class:`.SemanticSegmentationConfig`, and :class:`.ObjectDetectionConfig` API docs have more information on configuring pipelines.

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

Configs
^^^^^^^

A :class:`.Config` is a Raster Vision class based on the pydantic :class:`.BaseModel` class that allows various kinds of configurations to be stored in a systematic, typed, validatable, and serializable way. Most of these also implement a :meth:`build() <.Config.build>` method that allows the corresponding object to be created based on the configuration. For example, :meth:`.RasterioSourceConfig.build` builds a :class:`.RasterioSource` object.

.. note::

   **Configs and backward compatibility**

   Another crucial role that ``Configs`` play is enabling backward compatibility. Suppose you trained a model and stored it in a :ref:`model-bundle <bundle command>` using an older version of Raster Vision, and now want to use that bundle with a newer version of Raster Vision installed. This can be a problem if the specification of any ``Configs`` has changed between the two versions (e.g. if a field was removed or renamed), which means the newer version will not be able to deserialize the older pipeline config stored in the bundle. 

   Raster Vision solves this issue by associating each Raster Vision plugin with a version number (this is distinct from the Python package version) and providing a config-upgrader mechanism. You can define an upgrader function that takes as input the serialized config dict and a version number and modifies the dict in such a way that makes it compatible with the current version. This function is called multiple times for each config--once for each version number, from zero to the current version. An example upgrader function is shown below.

   .. code-block:: python

      def rs_config_upgrader(cfg_dict: dict, version: int) -> dict:
         if version == 6:
            # removed in version 7
            if cfg_dict.get('extent_crop') is not None:
                  raise ConfigError('RasterSourceConfig.extent_crop is deprecated.')
            try:
                  del cfg_dict['extent_crop']
            except KeyError:
                  pass
         elif version == 9:
            # renamed in version 10
            cfg_dict['bbox'] = cfg_dict.get('extent')
            try:
                  del cfg_dict['extent']
            except KeyError:
                  pass
         return cfg_dict

   This upgrader function can then be registered against the corresponding Config by passing it to the ``upgrader=`` keyword argument in :func:`.register_config` as shown below.

   .. code-block:: python

      @register_config('raster_source', upgrader=rs_config_upgrader)
      class RasterSourceConfig(Config):
         ...

The following table shows the corresponding ``Config`` counterparts for various commonly used classes.

.. currentmodule:: rastervision.core

.. list-table::
   :header-rows: 1

   * -  Class
     -  Config
     -  Notes
   * -  :class:`.Scene`
     -  :class:`.SceneConfig`
     -  :ref:`notes <scene>`
   * -  :class:`.RasterSource`


        - :class:`.RasterioSource`
        - :class:`.XarraySource`
        - :class:`.MultiRasterSource`
        - :class:`.TemporalMultiRasterSource`
        - :class:`.RasterizedSource`

     -  :class:`.RasterSourceConfig`

        - :class:`.RasterioSourceConfig`
        - :class:`.XarraySourceConfig`
        - :class:`.MultiRasterSourceConfig`
        - :class:`.RasterizedSourceConfig`

     -
   * -  :class:`.RasterTransformer`

        - :class:`.CastTransformer`
        - :class:`.MinMaxTransformer`
        - :class:`.NanTransformer`
        - :class:`.ReclassTransformer`
        - :class:`.RGBClassTransformer`
        - :class:`.StatsTransformer`

     -  :class:`.RasterTransformerConfig`

        - :class:`.CastTransformerConfig`
        - :class:`.MinMaxTransformerConfig`
        - :class:`.NanTransformerConfig`
        - :class:`.ReclassTransformerConfig`
        - :class:`.RGBClassTransformerConfig`
        - :class:`.StatsTransformerConfig`

     -
   * -  :class:`.VectorSource`

        - :class:`.GeoJSONVectorSource`

     -  :class:`.VectorSourceConfig`

        - :class:`.GeoJSONVectorSourceConfig`

     -
   * -  :class:`.VectorTransformer`

        - :class:`.BufferTransformer`
        - :class:`.ClassInferenceTransformer`
        - :class:`.ShiftTransformer`

     -  :class:`.VectorTransformerConfig`

        - :class:`.BufferTransformerConfig`
        - :class:`.ClassInferenceTransformerConfig`
        - :class:`.ShiftTransformerConfig`

     -
   * -  :class:`.LabelSource`

        - :class:`.ChipClassificationLabelSource`
        - :class:`.SemanticSegmentationLabelSource`
        - :class:`.ObjectDetectionLabelSource`

     -  :class:`.LabelSourceConfig`

        - :class:`.ChipClassificationLabelSourceConfig`
        - :class:`.SemanticSegmentationLabelSourceConfig`
        - :class:`.ObjectDetectionLabelSourceConfig`

     -
   * -  :class:`.LabelStore`

        - :class:`.ChipClassificationGeoJSONStore`
        - :class:`.ObjectDetectionGeoJSONStore`
        - :class:`.SemanticSegmentationLabelStore`

     -  :class:`.LabelStoreConfig`

        - :class:`.ChipClassificationGeoJSONStoreConfig`
        - :class:`.ObjectDetectionGeoJSONStoreConfig`
        - :class:`.SemanticSegmentationLabelStoreConfig`

     -  :ref:`notes <label store>`
   * -  :class:`.Analyzer`

        - :class:`.StatsAnalyzer`

     -  :class:`.AnalyzerConfig`

        - :class:`.StatsAnalyzerConfig`

     -  :ref:`notes <analyzer>`
   * -  :class:`.Evaluator`

        - :class:`.ChipClassificationEvaluator`
        - :class:`.SemanticSegmentationEvaluator`
        - :class:`.ObjectDetectionEvaluator`

     -  :class:`.EvaluatorConfig`

        - :class:`.ChipClassificationEvaluatorConfig`
        - :class:`.SemanticSegmentationEvaluatorConfig`
        - :class:`.ObjectDetectionEvaluatorConfig`

     -  :ref:`notes <evaluator>`

.. _backend:

Backend
^^^^^^^

.. currentmodule:: rastervision.core

:class:`RVPipelines <rv_pipeline.rv_pipeline.RVPipeline>` use a "backend" abstraction inspired by `Keras <https://keras.io/backend/>`_, which makes it easier to customize the code for building and training models (including using Raster Vision with arbitrary deep learning libraries).
Each backend is a subclass of :class:`.Backend` and has methods for saving training chips, training models, and making predictions, and is configured with a :class:`~Backend <backend.backend_config.BackendConfig>`.

The :mod:`rastervision.pytorch_backend` plugin provides backends that are thin wrappers around the :mod:`rastervision.pytorch_learner` package, which does most of the heavy lifting of building and training models using `torch <https://pytorch.org/docs/stable/>`_ and `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`_. (Note that :mod:`rastervision.pytorch_learner` is decoupled from :mod:`rastervision.pytorch_backend` so that it can be used in conjunction with :mod:`rastervision.pipeline` to write arbitrary computer vision pipelines that have nothing to do with remote sensing.)

Here are the PyTorch backends:

.. currentmodule:: rastervision.pytorch_backend

* The :class:`.PyTorchChipClassification` backend trains classification models from `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`_.
* The :class:`.PyTorchObjectDetection` backend trains the Faster-RCNN model in `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`_.
* The :class:`.PyTorchSemanticSegmentation` backend trains the DeepLabV3 model in `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`_.

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

A :class:`.DatasetConfig` defines the `training, validation, and test splits <https://en.wikipedia.org/wiki/Training,_test,_and_validation_sets>`_ needed to train and evaluate a model. Each dataset split is a list of `SceneConfigs <SceneConfig>`_.

In our {{ tiny_spacenet }} example, we configured the dataset with single scenes, though more often in real use cases you would use multiple scenes per split:

.. literalinclude:: /../rastervision_pytorch_backend/rastervision/pytorch_backend/examples/tiny_spacenet.py
    :language: python
    :lines: 23-30
    :dedent:

.. _scene:

Scene
^^^^^

.. currentmodule:: rastervision.core.data

A :ref:`usage_scene` is configured via a :class:`.SceneConfig` which is composed of the following elements:

* *Imagery*: a :class:`.RasterSourceConfig` represents a large scene image, which can be made up of multiple sub-images or a single file.
* *Ground truth labels*: a :class:`.LabelSourceConfig` represents ground-truth task-specific labels.
* *Predicted labels* (Optional): a :class:`.LabelStoreConfig` specifies how to store and retrieve the predictions from a scene.
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

A :ref:`usage_label_store` is configured via a :class:`.LabelStoreConfig`.

In the {{ tiny_spacenet }} example, there is no explicit :class:`.LabelStore` configured on the validation scene, because it can be inferred from the type of :class:`.RVPipelineConfig` it is part of.
In the {{ isprs_potsdam }} example, the following code is used to explicitly create a :class:`.LabelStoreConfig` that, in turn, will be used to create a :class:`.LabelStore` that writes out the predictions in "RGB" format, where the color of each pixel represents the class, and predictions of class 0 (ie. car) are also written out as polygons.

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

For each computer vision task, there is an :class:`.Evaluator` (configured via the corresponding :class:`.EvaluatorConfig`) that computes metrics for a trained model. It does this by measuring the discrepancy between ground truth and predicted labels for a set of validation scenes. Typically, you won't need to explicitly configure any.
