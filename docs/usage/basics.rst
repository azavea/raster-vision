Basic concepts
==============

At a high-level, a typical machine learning workflow on geospatial data involves the following steps:

- Read geospatial data
- Train a model
- Make predictions
- Write predictions (as geospatial data)

Below, we describe various Raster Vision components that can be used to perform these steps.

Reading geospatial data
-----------------------

.. currentmodule:: rastervision.core.data

Raster Vision internally uses the following pipeline for reading geo-referenced data and coaxing it into a form usable for training computer vision models.

When using Raster Vision :ref:`as a library <usage_library>`, users generally do not need to deal with all the individual components to arrive at a working `GeoDataset`_ (see the tutorial on :doc:`tutorials/sampling_training_data`), but certainly can if needed.

.. image:: /img/usage-input.png
    :align: center

Below, we briefly describe each of the components shown in the diagram above.

RasterSource
~~~~~~~~~~~~

    Tutorial: :doc:`tutorials/reading_raster_data`

A :class:`~raster_source.raster_source.RasterSource` represents a source of raster data for a scene. It is used to retrieve small windows of raster data (or *chips*) from larger scenes. It can also be used of to subset image channels (i.e. bands) as well as do more complex transformations using :mod:`RasterTransformers <rastervision.core.data.raster_transformer>`. You can even combine bands from multiple sources using a :class:`~raster_source.multi_raster_source.MultiRasterSource`.

.. seealso::

    - :class:`~raster_source.rasterio_source.RasterioSource`
    - :class:`~raster_source.multi_raster_source.MultiRasterSource`
    - :class:`~raster_source.rasterized_source.RasterizedSource`
    - :class:`~raster_transformer.raster_transformer.RasterTransformer`

      - :class:`~raster_transformer.cast_transformer.CastTransformer`
      - :class:`~raster_transformer.min_max_transformer.MinMaxTransformer`
      - :class:`~raster_transformer.nan_transformer.NanTransformer`
      - :class:`~raster_transformer.reclass_transformer.ReclassTransformer`
      - :class:`~raster_transformer.rgb_class_transformer.RGBClassTransformer`
      - :class:`~raster_transformer.stats_transformer.StatsTransformer`

VectorSource
~~~~~~~~~~~~

    Tutorial: :doc:`tutorials/reading_vector_data`

Annotations for geospatial data are often represented as vector data such as polygons and lines. A :class:`~vector_source.vector_source.VectorSource` is Raster Vision's abstraction for a vector data reader. Just like :class:`RasterSources <raster_source.raster_source.RasterSource>`, :class:`VectorSources <vector_source.vector_source.VectorSource>` also allow transforming the data using :mod:`VectorTransformers <rastervision.core.data.vector_transformer>`.

.. seealso::

    - :class:`~vector_source.geojson_vector_source.GeoJSONVectorSource`
    - :class:`~raster_source.rasterized_source.RasterizedSource`
    - :class:`~vector_transformer.vector_transformer.VectorTransformer`

      - :class:`~vector_transformer.buffer_transformer.BufferTransformer`
      - :class:`~vector_transformer.class_inference_transformer.ClassInferenceTransformer`
      - :class:`~vector_transformer.shift_transformer.ShiftTransformer`

LabelSource
~~~~~~~~~~~

    Tutorial: :doc:`tutorials/reading_labels`

A :class:`~label_source.label_source.LabelSource` interprets the data read by raster or vector sources into a form suitable for machine learning. They can be queried for the labels that lie within a window and are used for creating training chips, as well as providing ground truth labels for evaluation against validation scenes. There are different implementations available for :class:`chip classification <label_source.chip_classification_label_source.ChipClassificationLabelSource>`, :class:`semantic segmentation <label_source.semantic_segmentation_label_source.SemanticSegmentationLabelSource>`, and :class:`object detection <label_source.object_detection_label_source.ObjectDetectionLabelSource>`.

.. seealso::

    - :class:`~label_source.chip_classification_label_source.ChipClassificationLabelSource`
    - :class:`~label_source.semantic_segmentation_label_source.SemanticSegmentationLabelSource`
    - :class:`~label_source.object_detection_label_source.ObjectDetectionLabelSource`

Scene
~~~~~

    Tutorial: :doc:`tutorials/scenes_and_aois`

A :class:`~scene.Scene` is essentially a combination of a `RasterSource`_ and a `LabelSource`_ along with an optional AOI which can be specified as one or more polygons.

It can also

- hold a `LabelStore`_; this is useful for evaluating predictions against ground truth labels
- just have a `RasterSource`_ without a `LabelSource`_ or `LabelStore`_; this can be useful if you want to turn it into a dataset to be used for unsupervised or self-supervised learning

Scenes can also be more `conveniently initialized <tutorials/scenes_and_aois.ipynb#Easier-initialization>`_ using the factory functions defined in :mod:`rastervision.core.data.utils.factory`.

.. _usage_geodataset:

GeoDataset
~~~~~~~~~~

    Tutorial: :doc:`tutorials/sampling_training_data`

.. currentmodule:: rastervision.pytorch_learner

A :class:`~dataset.dataset.GeoDataset` (provided by Raster Vision's :mod:`~rastervision.pytorch_learner` plugin) is a `PyTorch-compatible dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`__ that can readily be wrapped into a `DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__ and used by any PyTorch training code. Raster Vision provides a :class:`~learner.Learner` class for training models, but you can also use GeoDatasets with either your own custom training code, or with a 3rd party library like `PyTorch Lightning <https://www.pytorchlightning.ai/>`__.

.. seealso::

    - :class:`~dataset.dataset.AlbumentationsDataset` (base dataset class)
    - :class:`~dataset.dataset.GeoDataset`

      - :class:`~dataset.dataset.SlidingWindowGeoDataset`

        - :class:`~dataset.classification_dataset.ClassificationSlidingWindowGeoDataset`
        - :class:`~dataset.semantic_segmentation_dataset.SemanticSegmentationSlidingWindowGeoDataset`
        - :class:`~dataset.object_detection_dataset.ObjectDetectionSlidingWindowGeoDataset`
        - :class:`~dataset.regression_dataset.RegressionSlidingWindowGeoDataset`

      - :class:`~dataset.dataset.RandomWindowGeoDataset`

        - :class:`~dataset.classification_dataset.ClassificationRandomWindowGeoDataset`
        - :class:`~dataset.semantic_segmentation_dataset.SemanticSegmentationRandomWindowGeoDataset`
        - :class:`~dataset.object_detection_dataset.ObjectDetectionRandomWindowGeoDataset`
        - :class:`~dataset.regression_dataset.RegressionRandomWindowGeoDataset`


Training a model
----------------

.. image:: /img/usage-train.png
    :align: center

Learner
~~~~~~~

    Tutorial: :doc:`tutorials/train`

.. currentmodule:: rastervision.pytorch_learner

Raster Vision's :mod:`~rastervision.pytorch_learner` plugin provides a :class:`~learner.Learner` class that encapsulates the entire training process. It is highly configurable. You can either fill out a :class:`~learner_config.LearnerConfig` and have the :class:`~learner.Learner` set everything up (datasets, model, loss, optimizers, etc.) for you, or you can pass in your own models, datasets, etc. and have the :class:`~learner.Learner` use them instead.

The main output of the :class:`~learner.Learner` is a trained model. This is available as a ``last-model.pth`` file which is a serialized dictionary of model weights that can be loaded into a model via ``model.load_state_dict(torch.load('last-model.pth'))``. You can also make the :class:`~learner.Learner` output a "model-bundle" (via :meth:`~learner.Learner.save_model_bundle`), which outputs a zip file containing the model weights as well as a config file that can be used to re-create the :class:`~learner.Learner` via :meth:`~learner.Learner.from_model_bundle`.

There are Learner subclasses for :class:`chip classification <classification_learner.ClassificationLearner>`, :class:`semantic segmentation <semantic_segmentation_learner.SemanticSegmentationLearner>`, :class:`object detection <object_detection_learner.ObjectDetectionLearner>`, and :class:`regression <regression_learner.RegressionLearner>`.

.. note::

    The :class:`Learners <learner.Learner>` are not limited to :class:`GeoDatasets <dataset.dataset.GeoDataset>` and can work with any PyTorch-compatible image dataset. In fact, :mod:`~rastervision.pytorch_learner` also provides an :class:`~dataset.dataset.ImageDataset` class for dealing with non-geospatial datasets.

.. seealso::

    - :class:`~classification_learner.ClassificationLearner`
    - :class:`~semantic_segmentation_learner.SemanticSegmentationLearner`
    - :class:`~object_detection_learner.ObjectDetectionLearner`
    - :class:`~regression_learner.RegressionLearner`
    - :class:`~dataset.dataset.ImageDataset`

      - :class:`~dataset.classification_dataset.ClassificationImageDataset`
      - :class:`~dataset.semantic_segmentation_dataset.SemanticSegmentationImageDataset` (and :class:`~dataset.semantic_segmentation_dataset.SemanticSegmentationDataReader`)
      - :class:`~dataset.object_detection_dataset.ObjectDetectionImageDataset` (and :class:`~dataset.object_detection_dataset.CocoDataset`)
      - :class:`~dataset.regression_dataset.RegressionImageDataset` (and :class:`~dataset.regression_dataset.RegressionDataReader`)

Making predictions and saving them
----------------------------------

    Tutorial: :doc:`tutorials/pred_and_eval_ss`

.. currentmodule:: rastervision.pytorch_learner

.. image:: /img/usage-pred.png
    :align: center

Having trained a model, you would naturally want to use it to make predictions on new scenes. The usual workflow for this is:

1. Instantiate a :class:`~learner.Learner` form a model-bundle (via :meth:`~learner.Learner.from_model_bundle`)
2. Instantiate the appropriate :class:`~dataset.dataset.SlidingWindowGeoDataset` subclass e.g. :class:`~dataset.semantic_segmentation_dataset.SemanticSegmentationSlidingWindowGeoDataset` (can be done easily using the convenience method :meth:`~dataset.semantic_segmentation_dataset.SemanticSegmentationSlidingWindowGeoDataset.from_uris`)
3. Pass the :class:`~dataset.dataset.SlidingWindowGeoDataset` to :meth:`Learner.predict_dataset() <learner.Learner.predict_dataset>`
4. Convert predictions into the appropriate `Labels`_ subclass e.g. :class:`~rastervision.core.data.label.semantic_segmentation_labels.SemanticSegmentationLabels` (via :meth:`~rastervision.core.data.label.semantic_segmentation_labels.SemanticSegmentationLabels.from_predictions`)
5. Save the `Labels`_ to file (via :meth:`~rastervision.core.data.label.semantic_segmentation_labels.SemanticSegmentationLabels.save`)

   - Alternatively, you can Instantiate an appropriate `LabelStore`_ subclass and pass the :class:`~rastervision.core.data.label.label.Labels` to :meth:`LabelStore.save() <rastervision.core.data.label_store.label_store.LabelStore.save>`

Labels
~~~~~~

.. currentmodule:: rastervision.core.data

The :class:`~label.labels.Labels` class is an in-memory representation of labels. It can represent both ground truth labels or model predictions.

.. seealso::

    - :class:`~label.chip_classification_labels.ChipClassificationLabels`
    - :class:`~label.object_detection_labels.ObjectDetectionLabels`
    - :class:`~label.semantic_segmentation_labels.SemanticSegmentationLabels`

      - :class:`~label.semantic_segmentation_labels.SemanticSegmentationDiscreteLabels`
      - :class:`~label.semantic_segmentation_labels.SemanticSegmentationSmoothLabels`

LabelStore
~~~~~~~~~~

.. currentmodule:: rastervision.core.data

A :class:`~label_store.label_store.LabelStore` abstracts away the writing of :class:`~label.labels.Labels` to file. It can also be used to read previously written predictions back as :class:`~label.labels.Labels`; this is useful for evaluating predictions.

.. seealso::

    - :class:`~label_store.chip_classification_geojson_store.ChipClassificationGeoJSONStore`
    - :class:`~label_store.object_detection_geojson_store.ObjectDetectionGeoJSONStore`
    - :class:`~label_store.semantic_segmentation_label_store.SemanticSegmentationLabelStore`
