.. rst-class:: hide-header

.. currentmodule:: rastervision

Welcome to Raster Vision
========================

|

.. image:: _static/raster-vision-logo.png
    :align: center
    :scale: 50%
    :target: https://rastervision.io

|

**Raster Vision** is an open source framework for engineers building computer
vision models on satellite, aerial, and other large imagery sets (including
oblique drone imagery). It allows for engineers to quickly and repeatably
configure *experiments* that go through core components of a machine learning
workflow: analyzing training data, creating training chips, training models,
creating predictions, evaluating models, and bundling the model files and
configuration for easy deployment.

Raster Vision workflows begin when you have a set of images and training data,
optionally with AOIs that describe where the images are labeled. Raster Vision
workflows end with a packaged model and configuration that allows you to
easily utilize models in various  deployment situations. Inside the Raster Vision
workflow, there's the proccess of running multiple experiments to find the best model
or models to deploy.

.. image:: _static/overview-raster-vision-workflow.png
    :align: center
    :scale: 80%

The process of running experiments includes executing workflows that perform the following
commands

.. click:example::

  import rastervision as rv

  class FooExperimentSet(rv.ExperimentSet):
      def exp_main(self, tmp_dir):
          task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                              .with_chip_size(200) \
                              .with_classes({
                                  'car': (1, 'red'),
                                  'building': (2, 'blue'),
                                  'background': (3, 'black')
                              }) \
                              .build()

          backend = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                                    .with_task(task) \
                                    .with_model_defaults(rv.RESNET50_IMAGENET) \
                                    .with_num_epochs(40) \
                                    .build()

          label_source = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
                                             .with_uri(training_label_uri) \
                                             .with_ioa_thresh(0.5) \
                                             .with_pick_min_class_id(True) \
                                             .with_background_class_id(3) \
                                             .with_infer_cells(True) \
                                             .build()

          raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                               .with_uri(train_img_uri) \
                                               .with_channel_order([0, 1, 2]) \
                                               .with_stats_transformer() \
                                               .build()

          train_scene = rv.SceneConfig.builder() \
                                      .with_task(task) \
                                      .with_id('training_scene') \
                                      .with_raster_source(raster_source) \
                                      .with_label_source(label_source) \
                                      .build()

          val_scene = rv.SceneConfig.builder() \
                                    .with_task(task) \
                                    .with_id('val_scene') \
                                    .with_raster_source(val_image_uri) \
                                    .build()

          dataset = rv.DatasetConfig.builder() \
                                    .with_train_scene(train_scene) \
                                    .with_validation_scene(val_scene) \
                                    .build()

          experiment = rv.ExperimentConfig.builder() \
                                          .with_id('foo-experiment') \
                                          .with_root_uri(root_uri) \
                                          .with_task(task) \
                                          .with_backend(backend) \
                                          .with_dataset(dataset) \
                                          .with_stats_analyzer() \
                                          .build()

          return experiment


  if __name__ == '__main__':
      rv.main()

Raster Vision uses a ``unittest``-like method for executing experiments. For instance, if the
above was defined in `foo.py`, you could run the experiment locally by running:

.. code:: shell

   > rastervision run -p foo.py


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
   usage
   experiments
   commands
   runners
   predictor
   cli
   codebase
   extending
   qgis
   contributing

API Reference
-------------

If you are looking for information on a specific function, class, or
method, this part of the documentation is for you.

.. toctree::
   :maxdepth: 2

   api
