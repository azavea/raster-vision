.. rst-class:: hide-header

.. currentmodule:: rastervision

|

.. image:: _static/raster-vision-logo-index.png
    :align: center
    :target: https://rastervision.io

|

**Raster Vision** is an open source framework for Python developers building computer
vision models on satellite, aerial, and other large imagery sets (including
oblique drone imagery). It allows for engineers to quickly and repeatably
configure *experiments* that go through core components of a machine learning
workflow: analyzing training data, creating training chips, training models,
creating predictions, evaluating models, and bundling the model files and
configuration for easy deployment.

Raster Vision workflows begin when you have a set of images and training data,
optionally with Areas of Interest (AOIs) that describe where the images are labeled. Raster Vision
workflows end with a packaged model and configuration that allows you to
easily utilize models in various  deployment situations. Inside the Raster Vision
workflow, there's the process of running multiple experiments to find the best model
or models to deploy.

.. image:: _static/overview-raster-vision-workflow.png
    :align: center

The process of running experiments includes executing workflows that perform the following
commands:

* **ANALYZE**: Gather dataset-level statistics and metrics for use in downstream processes.
* **CHIP**: Create training chips from a variety of image and label sources.
* **TRAIN**: Train a model using a variety of "backends" such as TensorFlow or Keras.
* **PREDICT**: Make predictions using trained models on validation and test data.
* **EVAL**: Derive evaluation metrics such as F1 score, precision and recall against the model's predictions on validation datasets.
* **BUNDLE**: Bundle the trained model into a :ref:`predict package`, which can be deployed in batch processes, live servers, and other workflows.

Experiments are configured using a fluent builder pattern that makes configuration easy to read, reuse
and maintain.

.. click:example::

   # tiny_spacenet.py

   import rastervision as rv

   class TinySpacenetExperimentSet(rv.ExperimentSet):
       def exp_main(self):
           base_uri = ('https://s3.amazonaws.com/azavea-research-public-data/'
                       'raster-vision/examples/spacenet')
           train_image_uri = '{}/RGB-PanSharpen_AOI_2_Vegas_img205.tif'.format(base_uri)
           train_label_uri = '{}/buildings_AOI_2_Vegas_img205.geojson'.format(base_uri)
           val_image_uri = '{}/RGB-PanSharpen_AOI_2_Vegas_img25.tif'.format(base_uri)
           val_label_uri = '{}/buildings_AOI_2_Vegas_img25.geojson'.format(base_uri)

           task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                               .with_chip_size(512) \
                               .with_classes({
                                   'building': (1, 'red')
                               }) \
                               .with_chip_options(neg_ratio=1.0,
                                                  ioa_thresh=0.8) \
                               .with_predict_options(merge_thresh=0.1,
                                                     score_thresh=0.5) \
                               .build()

           backend = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                                     .with_task(task) \
                                     .with_debug(True) \
                                     .with_batch_size(8) \
                                     .with_num_steps(5) \
                                     .with_model_defaults(rv.SSD_MOBILENET_V2_COCO)  \
                                     .build()

           train_raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                                      .with_uri(train_image_uri) \
                                                      .with_stats_transformer() \
                                                      .build()

           train_scene =  rv.SceneConfig.builder() \
                                        .with_task(task) \
                                        .with_id('train_scene') \
                                        .with_raster_source(train_raster_source) \
                                        .with_label_source(train_label_uri) \
                                        .build()

           val_raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                                    .with_uri(val_image_uri) \
                                                    .with_stats_transformer() \
                                                    .build()

           val_scene = rv.SceneConfig.builder() \
                                     .with_task(task) \
                                     .with_id('val_scene') \
                                     .with_raster_source(val_raster_source) \
                                     .with_label_source(val_label_uri) \
                                     .build()

           dataset = rv.DatasetConfig.builder() \
                                     .with_train_scene(train_scene) \
                                     .with_validation_scene(val_scene) \
                                     .build()

           experiment = rv.ExperimentConfig.builder() \
                                           .with_id('tiny-spacenet-experiment') \
                                           .with_root_uri('/opt/data/rv') \
                                           .with_task(task) \
                                           .with_backend(backend) \
                                           .with_dataset(dataset) \
                                           .with_stats_analyzer() \
                                           .build()

           return experiment


   if __name__ == '__main__':
       rv.main()

Raster Vision uses a ``unittest``-like method for executing experiments. For instance, if the
above was defined in `tiny_spacenet.py`, with the proper setup you could run the experiment
on AWS Batch by running:

.. code:: shell

   > rastervision run aws_batch -p tiny_spacenet.py

See the :ref:`quickstart` for a more complete description of using this example.


.. _documentation:

Documentation
=============

This part of the documentation guides you through all of the library's
usage patterns.

.. toctree::
   :maxdepth: 2

   why
   quickstart
   setup
   experiments
   commands
   runners
   predictor
   cli
   misc
   codebase
   plugins
   qgis
   contributing

API Reference
-------------

If you are looking for information on a specific function, class, or
method, this part of the documentation is for you.

.. toctree::
   :maxdepth: 10

   api
