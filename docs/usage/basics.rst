Basic Concepts
==============

At a high-level, a typical machine learning workflow for geospatial data involves the following steps:

- Read geospatial data
- Train a model
- Make predictions
- Write predictions (as geospatial data)

Below, we describe various Raster Vision components that can be used to perform these steps.

Reading geospatial data
-----------------------

.. currentmodule:: rastervision.core.data

Raster Vision internally uses the following pipeline for reading geo-referenced data and coaxing it into a form suitable for training computer vision models.

When using Raster Vision :ref:`as a library <usage_library>`, users generally do not need to deal with all the individual components to arrive at a working `GeoDataset`_ (see the tutorial on :doc:`tutorials/sampling_training_data`), but certainly can if needed.

.. image:: /img/usage-input.png
    :align: center
    :class: only-light

.. image:: /img/usage-input.png
    :align: center
    :class: only-dark

Below, we briefly describe each of the components shown in the diagram above.

.. _usage_raster_source:

RasterSource
~~~~~~~~~~~~

    Tutorial: :doc:`tutorials/reading_raster_data`

A :class:`.RasterSource` represents a source of raster data for a scene. It is used to retrieve small windows of raster data (or *chips*) from larger scenes. It can also be used to subset image channels (i.e. bands) as well as do more complex transformations using :mod:`RasterTransformers <rastervision.core.data.raster_transformer>`. You can even combine bands from multiple sources using a :class:`.MultiRasterSource` or stack images from sources in a time-series using a :class:`.TemporalMultiRasterSource`.

.. seealso::

    - :class:`.RasterioSource`
    - :class:`.XarraySource`
    - :class:`.MultiRasterSource`
    - :class:`.TemporalMultiRasterSource`
    - :class:`.RasterizedSource`
    - :class:`.RasterTransformer`

      - :class:`.CastTransformer`
      - :class:`.MinMaxTransformer`
      - :class:`.NanTransformer`
      - :class:`.ReclassTransformer`
      - :class:`.RGBClassTransformer`
      - :class:`.StatsTransformer`

VectorSource
~~~~~~~~~~~~

    Tutorial: :doc:`tutorials/reading_vector_data`

Annotations for geospatial data are often represented as vector data such as polygons and lines. A :class:`.VectorSource` is Raster Vision's abstraction for a vector data reader. Just like :class:`RasterSources <raster_source.raster_source.RasterSource>`, :class:`VectorSources <vector_source.vector_source.VectorSource>` also allow transforming the data using :mod:`VectorTransformers <rastervision.core.data.vector_transformer>`.

.. seealso::

    - :class:`.GeoJSONVectorSource`
    - :class:`.RasterizedSource`
    - :class:`.VectorTransformer`

      - :class:`.BufferTransformer`
      - :class:`.ClassInferenceTransformer`
      - :class:`.ShiftTransformer`

LabelSource
~~~~~~~~~~~

    Tutorial: :doc:`tutorials/reading_labels`

A :class:`.LabelSource` interprets the data read by raster or vector sources into a form suitable for machine learning. They can be queried for the labels that lie within a window and are used for creating training chips, as well as providing ground truth labels for evaluation against model predictions. There are different implementations available for :class:`chip classification <label_source.chip_classification_label_source.ChipClassificationLabelSource>`, :class:`semantic segmentation <label_source.semantic_segmentation_label_source.SemanticSegmentationLabelSource>`, and :class:`object detection <label_source.object_detection_label_source.ObjectDetectionLabelSource>`.

.. seealso::

    - :class:`.ChipClassificationLabelSource`
    - :class:`.SemanticSegmentationLabelSource`
    - :class:`.ObjectDetectionLabelSource`

.. _usage_scene:

Scene
~~~~~

    Tutorial: :doc:`tutorials/scenes_and_aois`

A :class:`.Scene` is essentially a combination of a `RasterSource`_ and a `LabelSource`_ along with an optional AOI which can be specified as one or more polygons.

It can also

- hold a `LabelStore`_; this is useful for evaluating predictions against ground truth labels
- just have a `RasterSource`_ without a `LabelSource`_ or `LabelStore`_; this can be useful if you want to turn it into a dataset to be used for unsupervised or self-supervised learning

Scenes can also be more `conveniently initialized <tutorials/scenes_and_aois.ipynb#Easier-initialization>`_ using the factory functions defined in :mod:`rastervision.core.data.utils.factory`.

.. _usage_geodataset:

GeoDataset
~~~~~~~~~~

    Tutorial: :doc:`tutorials/sampling_training_data`

.. currentmodule:: rastervision.pytorch_learner

A :class:`.GeoDataset` (provided by Raster Vision's :mod:`~rastervision.pytorch_learner` plugin) is a `PyTorch-compatible dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`__ that can readily be wrapped into a `DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__ and used by any PyTorch training code. Raster Vision provides a :class:`.Learner` class for training models, but you can also use GeoDatasets with either your own custom training code, or with a 3rd party library like :doc:`PyTorch Lightning <tutorials/lightning_workflow>`.

.. seealso::

    - :class:`.AlbumentationsDataset` (base dataset class)
    - :class:`.GeoDataset`

      - :class:`.SlidingWindowGeoDataset`

        - :class:`.ClassificationSlidingWindowGeoDataset`
        - :class:`.SemanticSegmentationSlidingWindowGeoDataset`
        - :class:`.ObjectDetectionSlidingWindowGeoDataset`
        - :class:`.RegressionSlidingWindowGeoDataset`

      - :class:`.RandomWindowGeoDataset`

        - :class:`.ClassificationRandomWindowGeoDataset`
        - :class:`.SemanticSegmentationRandomWindowGeoDataset`
        - :class:`.ObjectDetectionRandomWindowGeoDataset`
        - :class:`.RegressionRandomWindowGeoDataset`


Training a model
----------------

.. image:: /img/usage-train.png
    :align: center
    :class: only-light

.. image:: /img/usage-train.png
    :align: center
    :class: only-dark

Learner
~~~~~~~

    Tutorial: :doc:`tutorials/train`

.. currentmodule:: rastervision.pytorch_learner

Raster Vision's :mod:`~rastervision.pytorch_learner` plugin provides a :class:`.Learner` class that encapsulates the entire training process. It is highly configurable. You can either fill out a :class:`.LearnerConfig` and have the :class:`.Learner` set everything up (datasets, model, loss, optimizers, etc.) for you, or you can pass in your own models, datasets, etc. and have the :class:`.Learner` use them instead.

The main output of the :class:`.Learner` is a trained model. This is available as a ``last-model.pth`` file which is a serialized dictionary of model weights that can be loaded into a model via

.. code-block:: python

    model.load_state_dict(torch.load('last-model.pth'))

You can also make the :class:`.Learner` output a "model-bundle" (via :meth:`~learner.Learner.save_model_bundle`), which outputs a zip file containing the model weights as well as a config file that can be used to re-create the :class:`.Learner` via :meth:`~learner.Learner.from_model_bundle`.

There are Learner subclasses for :class:`chip classification <classification_learner.ClassificationLearner>`, :class:`semantic segmentation <semantic_segmentation_learner.SemanticSegmentationLearner>`, :class:`object detection <object_detection_learner.ObjectDetectionLearner>`, and :class:`regression <regression_learner.RegressionLearner>`.

.. note::

    The :class:`Learners <learner.Learner>` are not limited to :class:`GeoDatasets <dataset.dataset.GeoDataset>` and can work with any PyTorch-compatible image dataset. In fact, :mod:`~rastervision.pytorch_learner` also provides an :class:`.ImageDataset` class for dealing with non-geospatial datasets.

.. seealso::

    - :class:`.ClassificationLearner`
    - :class:`.SemanticSegmentationLearner`
    - :class:`.ObjectDetectionLearner`
    - :class:`.RegressionLearner`
    - :class:`.ImageDataset`

      - :class:`.ClassificationImageDataset`
      - :class:`.SemanticSegmentationImageDataset` (and :class:`.SemanticSegmentationDataReader`)
      - :class:`.ObjectDetectionImageDataset` (and :class:`.CocoDataset`)
      - :class:`.RegressionImageDataset` (and :class:`.RegressionDataReader`)

Making predictions and saving them
----------------------------------

    Tutorial: :doc:`tutorials/pred_and_eval_ss`

.. currentmodule:: rastervision.pytorch_learner

.. image:: /img/usage-pred.png
    :align: center
    :class: only-light

.. image:: /img/usage-pred.png
    :align: center
    :class: only-dark

Having trained a model, you would naturally want to use it to make predictions on new scenes. The usual workflow for this is:

1. Instantiate a :class:`.Learner` form a model-bundle (via :meth:`~learner.Learner.from_model_bundle`)
2. Instantiate the appropriate :class:`.SlidingWindowGeoDataset` subclass e.g. :class:`.SemanticSegmentationSlidingWindowGeoDataset` (can be done easily using the convenience method :meth:`~dataset.semantic_segmentation_dataset.SemanticSegmentationSlidingWindowGeoDataset.from_uris`)
3. Pass the :class:`.SlidingWindowGeoDataset` to :meth:`Learner.predict_dataset() <learner.Learner.predict_dataset>`
4. Convert predictions into the appropriate `Labels`_ subclass e.g. :class:`.SemanticSegmentationLabels` (via :meth:`~rastervision.core.data.label.semantic_segmentation_labels.SemanticSegmentationLabels.from_predictions`)
5. Save the `Labels`_ to file (via :meth:`~rastervision.core.data.label.semantic_segmentation_labels.SemanticSegmentationLabels.save`)

   - Alternatively, you can Instantiate an appropriate `LabelStore`_ subclass and pass the :class:`.Labels` to :meth:`LabelStore.save() <rastervision.core.data.label_store.label_store.LabelStore.save>`

Labels
~~~~~~

.. currentmodule:: rastervision.core.data

The :class:`.Labels` class is an in-memory representation of labels. It can represent both ground truth labels and model predictions.

.. seealso::

    - :class:`.ChipClassificationLabels`
    - :class:`.ObjectDetectionLabels`
    - :class:`.SemanticSegmentationLabels`

      - :class:`.SemanticSegmentationDiscreteLabels`
      - :class:`.SemanticSegmentationSmoothLabels`

.. _usage_label_store:

LabelStore
~~~~~~~~~~

.. currentmodule:: rastervision.core.data

A :class:`.LabelStore` abstracts away the writing of :class:`.Labels` to file. It can also be used to read previously written predictions back as :class:`.Labels` which is useful for evaluating predictions.

.. seealso::

    - :class:`.ChipClassificationGeoJSONStore`
    - :class:`.ObjectDetectionGeoJSONStore`
    - :class:`.SemanticSegmentationLabelStore`
