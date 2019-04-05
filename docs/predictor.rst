Making Predictions (Inference)
==============================

A major focus of Raster Vision is to generate models that can quickly be used to
predict, or run inference, on new imagery. To accomplish this, the last step in the chain of commands
that comprise an experiment is the ``BUNDLE`` command, which generates a "predict package".
This predict package contains all the necessary model files and configuration to
make predictions using the model that was trained by an experiment.

How to make predictions with models trained by Raster Vision
------------------------------------------------------------

With a predict package, we can call the :ref:`predict cli command` command from the
command line client, or use the :ref:`predictor api` class to generate
predictions from a predict package directly from Python code.

Using the command line tool loads the model and saves the predictions for a single scene.
If you need to call this for a large number of scenes, consider using the ``Predictor`` programmatically, as this will allow you to load the model once and use it many times. This can
matter a lot if you want the time-to-prediction to be as fast as possible - the model
load time can be orders of magnitudes slower than the prediction time of a loaded model.

The ``Predictor`` class is the most flexible way to integrate Raster Vision models
into other systems, whether in large PySpark batch jobs or in web servers running
on GPU systems.

.. _predict package:

Predict Package
---------------

The predict package is a zip file containing the model file and the configuration necessary for
Raster Vision to use the model. The model file or files are specific to the backend: for
Keras, there's a single serialized Keras model file, and for TensorFlow there is the protobuf
serialized inference graph. But this is not all that is needed to create predictions. The
data that was trained on was potentially processed in specific ways by :ref:`raster transformer`,
and the model could have trained on a subset of bands dictated by the :ref:`raster source`.
We need to know about the :ref:`label store` that was used to serialize the predictions to GeoJSON, GeoTIFF, or something else. The prediction logic also needs to know which :ref:`Task` was used to apply any transformations that take raw model output and transform it to meaningful predictions.

The predict package holds all of this necessary information, so that a prediction call only needs
to know what imagery it is predicting against. This works generically over all models produced
by Raster Vision, without additional client considerations, and therefore abstracts away the specifics
of every model when considering how to deploy prediction software. Note that this means that by default, predictions will be made according to the configuration of the experiment that produced the predict package. Some of this configuration might be inappropriate for the new imagery (such as the ``channel_order``), and can be overridden by options to the :ref:`predict cli command` command.
