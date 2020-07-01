.. rst-class:: hide-header

.. currentmodule:: rastervision

|

.. image:: img/raster-vision-logo-index.png
    :align: center
    :target: https://rastervision.io

|

**Raster Vision** is an open source framework for Python developers building computer
vision models on satellite, aerial, and other large imagery sets (including
oblique drone imagery). There is built-in support for chip classification, object detection, and semantic segmentation using PyTorch.

.. image:: img/cv-tasks.png
    :align: center

Raster Vision allows engineers to quickly and repeatably
configure *pipelines* that go through core components of a machine learning
workflow: analyzing training data, creating training chips, training models,
creating predictions, evaluating models, and bundling the model files and
configuration for easy deployment.

.. image:: img/rv-pipeline-overview.png
    :align: center

The input to a Raster Vision pipeline is a set of images and training data,
optionally with Areas of Interest (AOIs) that describe where the images are labeled.
The output of a Raster Vision pipeline is a model bundle that allows you to
easily utilize models in various deployment scenarios.

The pipelines include running the following commands:

* **ANALYZE**: Gather dataset-level statistics and metrics for use in downstream processes.
* **CHIP**: Create training chips from a variety of image and label sources.
* **TRAIN**: Train a model using a "backend" such as PyTorch.
* **PREDICT**: Make predictions using trained models on validation and test data.
* **EVAL**: Derive evaluation metrics such as F1 score, precision and recall against the model's predictions on validation datasets.
* **BUNDLE**: Bundle the trained model and associated configuration into a :ref:`model bundle <model bundle>`, which can be deployed in batch processes, live servers, and other workflows.

Pipelines are configured using a compositional, programmatic approach that makes configuration easy to read, reuse, and maintain. Below, we show the ``tiny_spacenet`` example.

.. literalinclude:: ../rastervision_pytorch_backend/rastervision/pytorch_backend/examples/tiny_spacenet.py
   :language: python
   :caption: tiny_spacenet.py

Raster Vision uses a ``unittest``-like method for executing pipelines. For instance, if the
above was defined in `tiny_spacenet.py`, with the proper setup you could run the experiment
on AWS Batch by running:

.. code:: shell

   > rastervision run batch tiny_spacenet.py

See the :ref:`quickstart` for a more complete description of using this example.

.. _documentation:

Documentation
==================

This part of the documentation guides you through all of the library's
usage patterns.

.. toctree::
   :maxdepth: 2

   why
   quickstart
   setup
   cli
   pipelines
   architecture
   examples
   bootstrap
   cloudformation
   runners
   misc
   CONTRIBUTING
   release

.. _api:

Configuration API
==================

.. toctree::
   :maxdepth: 3

   api

CHANGELOG
===========

.. toctree::
   :maxdepth: 2

   changelog
