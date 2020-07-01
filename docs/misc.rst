Miscellaneous Topics
====================

.. _filesystem:

File Systems
--------------

The ``FileSystem`` architecture supports multiple file systems through an interface that is chosen by URI. There is built-in support for: local and HTTP file systems in the ``rastervision.pipeline`` package, AWS S3 in the ``rastervision.aws_s3`` plugin, and any file that can be opened using GDAL VSI in the ``rastervision.gdal_vsi`` plugin. Some file systems support read only (HTTP), while others are read/write. If you need to support other file storage systems, you can add new ``FileSystem`` classes via a plugin.

Viewing Tensorboard
-------------------

The PyTorch backends included in the ``rastervision.pytorch_backend`` plugin will start an instance of TensorBoard while training if ``log_tensorboard=True`` and ``run_tensorboard=True`` in the ``BackendConfig``.
To view TensorBoard, go to ``https://<domain>:6006/``. If you're running locally, then ``<domain>`` should
be ``localhost``, and if you are running remotely (for example AWS), <domain> is the public DNS of the machine running the training command. If running locally, make sure to forward port 6006 using the ``--tensorboard`` option to ``docker/run`` if you are using it. At the moment, basic metrics are logged each epoch, but more interesting visualization could be added in the future.

Transfer learning using models trained by RV
-----------------------------------------------

To use a model trained by Raster Vision for transfer learning or fine tuning, you can use output of the TRAIN command of the experiment as a pretrained model of further experiments. The ``last_model.pth`` model file in the ``train`` directory can be used as a pretrained model in a new pipeline. To do so, set the ``init_weights`` field to the model file in the ``ModelConfig`` in the new pipeline.

.. _model bundle:

Making Predictions with Model Bundles
----------------------------------------

To make predictions on new imagery, the :ref:`bundle <bundle command>` command generates a "model bundle" which can be used with the :ref:`predict cli command` command. This loads the model and saves the predictions for a single scene. If you need to call this for a large number of scenes, consider using the ``Predictor`` class programmatically, as this will allow you to load the model once and use it many times. This can
matter a lot if you want the time-to-prediction to be as fast as possible - the model load time can be orders of magnitudes slower than the prediction time of a loaded model.

The model bundle is a zip file containing the model weights and the configuration necessary for
Raster Vision to use the model. This configuration includes the configuration of the model architecture, how the training data was processed by :ref:`raster transformer`, the subset of bands used by the :ref:`raster source`, and potentially other things. The model bundle holds all of this necessary information, so that a prediction call only needs
to know what imagery it is predicting against.

This works generically over all models produced by Raster Vision, without additional client considerations, and therefore abstracts away the specifics
of every model when considering how to deploy prediction software. Note that this means that by default, predictions will be made according to the configuration of the pipeline that produced the model bundle. Some of this configuration might be inappropriate for the new imagery (such as the ``channel_order``), and can be overridden by options to the :ref:`predict cli command` command.
