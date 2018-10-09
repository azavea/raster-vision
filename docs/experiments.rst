Experiment Configuration
========================

Experiments are configured programmatically using a compositional API based on the fluent builder design pattern.

.. _experiment set:

Experiment Set
--------------

An experiment set is a set of related experiments and can be created by subclassing ExperimentSet. For each experiment, the class should have a method prefixed with ``exp_`` that returns an ExperimentConfig.

.. _experiment:

ExperimentConfig
----------------

.. image:: _static/experiments-experiment.png
    :align: center

.. _task:

An experiment is a sequence of commands that represents a machine learning workflow. For more information on the commands that comprise an experiment, see Commands.

Task
----

A ``Task`` is a computer vision task such as chip classification, object detection, or semantic segmentation. Each task is parameterized by its type, a chip size, and a set of classes.

Chip Classification
^^^^^^^^^^^^^^^^^^^

In chip classification, the goal is to divide the scene up into a grid of cells and classify each cell. This task is good for getting a rough idea of where certain objects are located, or where non-object "stuff" (such as grass) is located. It requires relatively low labeling effort, but also produces spatially coarse predictions. In our experience, this task trains the fastest, and is easiest to configure to get "decent" results.

Object Detection
^^^^^^^^^^^^^^^^

In object detection, the goal is to predict a bounding box and a class around each object of interest. This task requires higher labeling effort than chip classification, but has the ability to localize and individuate objects. Object detection models require more time to train and also struggle with objects that are very close together. In theory, it is straightforward to use object detection for counting objects.

Semantic Segmentation
^^^^^^^^^^^^^^^^^^^^^

In semantic segmentation, the goal is to predict the class of each pixel in a scene. This task requires the highest labeling effort, but also provides the most spatially precise predictions. Like object detection, these models take longer to train than chip classification models.

Future Tasks
^^^^^^^^^^^^

It is possible to add support for new tasks by extending the Task class. Some potential tasks to add are chip regression (goal: predict a number for each chip) and instance segmentation (goal: predict a segmentation mask for each individual object).

.. _backend:

Backend
-------

To avoid reinventing the wheel, Raster Vision relies on third-party libraries to implement core functionality around building and training models for the various computer vision tasks.
To maintain flexibility and avoid being tied to any one library, Raster Vision tasks interact with third-party libraries via a "backend" interface inspired by Keras.
https://keras.io/backend/
Each backend is a subclass of Backend and contains methods for translating between Raster Vision data structures and calls to a third-party library.

todo: pretrained models

Keras Classification
^^^^^^^^^^^^^^^^^^^^

For chip classification, the default backend is Keras Classification, which is a small, simple library for image classification using Keras. Currently, it only has support for ResNet50.

TensorFlow Object Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For object detection, the default backend is the Tensorflow Object Detection API. It supports a variety of object detection architectures such as SSD, Faster-RCNN, and RetinaNet with Mobilenet, ResNet, and Inception as base models.

TensorFlow DeepLab
^^^^^^^^^^^^^^^^^^

For object detection, the default backend is Tensorflow Deeplab. It has support for the Deeplab segmentation architecture with Mobilenet and Inception as base models.

Model Defaults
^^^^^^^^^^^^^^

For each backend, there is a list of  :ref:`model defaults` with a default configuration for each model architecture. Each default can be considered a good starting point for configuring that model.

Dataset
-------

A Dataset contains the training, validation, and test splits needed to train and evaluate a model. Each dataset split is a list of scenes. See https://en.wikipedia.org/wiki/Training,_test,_and_validation_sets

Scene
-----

.. image:: _static/experiments-scene.png
    :align: center

A scene represents an image, and associated labels, which are spatially-aligned task-specific annotations. A scene is composed of a RasterSource, a ground truth LabelSource and a prediction LabelStore.

.. _rastersource:

RasterSource
^^^^^^^^^^^^

A RasterSource represents a source of raster data for a scene. There are subclasses for different data sources including GeoTIFFSource, ImageSource (for non-georeferenced imagery such as .png files), and GeoJSONSource (for rasterized polygons and lines coming from a GeoJSON files).

.. _labelsource:

LabelSource
^^^^^^^^^^^

A LabelSource is an object that allows reading ground truth labels for a scene. There are subclasses for different tasks and data formats. They can be queried for the labels that lie within a window and are used for creating training chips.

.. _labelstore:

LabelStore
^^^^^^^^^^^

A `LabelStore` is an object that allows reading and writing predicted labels for a scene. There are subclasses for different tasks and data formats. They are used for saving predictions and then loading them during evaluation.

.. _rastertransformer:

Raster Transformers
^^^^^^^^^^^^^^^^^^^

A RasterTransformer is a mechanism for transforming raw raster data into a form that is more suitable for being fed into a model. For example, satellite imagery often contains more than three channels, but pretrained models trained on datasets like Imagenet only support three (RGB) input channels. In order to cope with this situation, we can use a RasterTransformer that selects three of the channels to utilize.

.. _augmentor:

Augmentors
^^^^^^^^^^

Data augmentation is a technique used to increase the effective size of a training dataset. It consists of transforming the images (and labels) using random shifts in position, rotation, zoom level, and color distribution. Each backend has its own ways of doing data augmentation inherited from its underlying third-party library, but some additional forms of data augmentation are implemented within Raster Vision as Augmentors.
For instance, there is a NodataAugmentor which adds blocks of NODATA values to images to learn to avoid making spurious predictions over NODATA regions.

.. _evaluator:

Evaluators
----------

For each task, there is an evaluator that computes metrics for a trained model. It does this by measuring the discrepancy between ground truth and predicted labels for a set of validation scenes.

.. _default provider:

Default Providers
-----------------
