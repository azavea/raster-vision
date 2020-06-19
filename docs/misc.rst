Miscellaneous Topics
====================

.. _filesystem:

FileSystems
-----------

The ``FileSystem`` architecture allows support of multiple file systems through an interface, that is chosen by URI. We currently support the local file system, AWS S3, and HTTP. Some filesystems support read only (HTTP), while others are read/write.

If you need to support other file storage systems, you can add new ``FileSystem`` classes via the plugin. We're happy to take contributions on new ``FileSystem`` support if it's generally useful!

Viewing Tensorboard
-------------------

The built-in backends will start an instance of TensorBoard while training.
To view TensorBoard, go to ``https://<domain>:6006/``. If you're running locally, then ``<domain>`` should
be ``localhost``, and if you are running remotely (for example AWS), <public_dns> is the public
DNS of the machine running the training command.

.. _model defaults:

Model Defaults
--------------

Model Defaults allow you to use a single key to set default attributes into backends instead of having to explicitly state them. This is useful for, say, using a key to refer to the pretrained model weights and hyperparameter configuration of various models. Each ``Backend`` can interpret its model defaults differently. For more information, see the `rastervision/backend/model_defaults.json <https://github.com/azavea/raster-vision/blob/0.11/rastervision/backend/model_defaults.json>`_ file.

You can set the model defaults to use a different JSON file, so that plugin backends can create model defaults or so that you can override the defaults provided by Raster Vision. See the :ref:`rv config section` Configuration Section for that config value.

Note that model defaults are only used for the Tensorflow-based backends.

TensorFlow Object Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a list of model defaults for use with the ``rv.TF_OBJECT_DETECTION`` backend.
They come from the TensorFlow Object Detection  project, and more information about what
each model is can be found in the `Tensorflow Object Detection Model Zoo <https://github.com/tensorflow/models/blob/63ecef1a3513b00c01f6aed75e178636746eff71/research/object_detection/g3doc/detection_model_zoo.md>`_ page.
These defaults include pretrained model weights and TensorFlow Object Detection ``pipeline.conf``
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

This is a list of model defaults for use with the ``rv.KERAS_CLASSIFICATION`` backend.
Keras Classification only supports one model for now, but more will be added in the future. The
pretrained weights come from `https://github.com/fchollet/deep-learning-models <https://github.com/fchollet/deep-learning-models>`_

* ``rv.RESNET50_IMAGENET``

Tensorflow DeepLab
^^^^^^^^^^^^^^^^^^

This is a list of model defaults for use with the ``rv.TF_DEEPLAB`` backend.
They come from the TensorFlow DeepLab project, and more information about
each model can be found in the `Tensorflow DeepLab Model Zoo <https://github.com/tensorflow/models/blob/63ecef1a3513b00c01f6aed75e178636746eff71/research/deeplab/g3doc/model_zoo.md>`_.
These defaults include pretrained model weights and backend configurations for the following models:

* ``rv.XCEPTION_65``
* ``rv.MOBILENET_V2``

Reusing models trained by Raster Vision
---------------------------------------

To use a model trained by Raster Vision for transfer learning or fine tuning, you can use output of the TRAIN command of the experiment as a pretrained model of further experiments. The files are listed per backend here:

* ``rv.PYTORCH_CHIP_CLASSIFICATION``: You can use the ``model`` file in the train command output as a pretrained model.
* ``rv.PYTORCH_SEMANTIC_SEGMENTATION``: You can use the ``model`` file in the train command output as a pretrained model.
* ``rv.PYTORCH_OBJECT_DETECTION``: You can use the ``model`` file in the train command output as a pretrained model.
* ``rv.KERAS_CLASSIFICATION``: You can use the ``model_weights.hdf5`` file in the train command output as a pretrained model.
* ``rv.TF_OBJECT_DETECTION``: Use the ``<experiment_id>.tar.gz`` that is in the train command output as a pretrained model. The default name of the file is the experiment ID, however you can change the backend configuration to use another name with the ``.with_fine_tune_checkpoint_name`` method.
* ``rv.TF_DEEPLAB``: Use the ``<experiment_id>.tar.gz`` that is in the TRAIN command output as a pretrained model. The default name of the file is the experiment ID, however you can change the backend configuration to use another name with the ``.with_fine_tune_checkpoint_name`` method.
