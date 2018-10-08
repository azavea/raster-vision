Miscellaneous Topics
====================

.. _filesystem:

FileSystems
-----------

TKTK

.. _model defaults:

Model Defaults
--------------

TensorFlow Object Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a list of models defaults for use with the ``rv.TF_OBJECT_DETECTION`` backend.
They come from the TensorFlow Object Detection  project, and more information about what
each model is can be found in the `Tensorflow Object Detection Model Zoo <https://github.com/tensorflow/models/blob/63ecef1a3513b00c01f6aed75e178636746eff71/research/object_detection/g3doc/detection_model_zoo.md>`_ page.
Default includes pretrained model weights and TensorFlow Object Detection ``pipeline.conf``
templates for the following models:

* ``rv.SSD_MOBILENET_V1_COCO``
* ``rv.SSD_MOBILENET_V2_COCO``
* ``rv.SSDLITE_MOBILENET_V2_COCO``
* ``rv.SSD_INCEPTION_V2_COCO``
* ``rv.FASTER_RCNN_INCEPTION_V2_COCO``
* ``rv.FASTER_RCNN_RESNET50_COCO``
* ``rv.RFCN_RESNET101_COCO``
* ``rv.FASTER_RCNN_RESNET101_COCO``
* ``rv.FASTER_RCNN_INCEPTION_RESNET_V2_ATROUS_COCO``
* ``rv.FASTER_RCNN_NAS``
* ``rv.MASK_RCNN_INCEPTION_RESNET_V2_ATROUS_COCO``
* ``rv.MASK_RCNN_INCEPTION_V2_COCO``
* ``rv.MASK_RCNN_RESNET101_ATROUS_COCO``
* ``rv.MASK_RCNN_RESNET50_ATROUS_COCO``

Keras Classification
^^^^^^^^^^^^^^^^^^^^

This is a list of models defaults for use with the ``rv.KERAS_CLASSIFICATION`` backend.
Keras Classification only supports one model for now, but more will be added in the future. The
pretrained weights come from `https://github.com/fchollet/deep-learning-models <https://github.com/fchollet/deep-learning-models>`_

* ``rv.RESNET50_IMAGENET``

Tensorflow DeepLab
^^^^^^^^^^^^^^^^^^

This is a list of models defaults for use with the ``rv.TF_DEEPLAB`` backend.
They come from the TensorFlow DeepLabl  project, and more information about what
each model is can be found in the `Tensorflow DeepLab Model Zoo <https://github.com/tensorflow/models/blob/63ecef1a3513b00c01f6aed75e178636746eff71/research/deeplab/g3doc/model_zoo.md>`_ page.
Default includes pretrained model weights and backend configuration:

* ``rv.XCEPTION_65``
* ``rv.MOBILENET_V2``
