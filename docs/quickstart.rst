.. _quickstart:

Quickstart
==========

In this Quickstart, we'll train a semantic segmentation model on `SpaceNet <https://spacenetchallenge.github.io/datasets/datasetHomePage.html>`_ data. Don't get too excited - we'll only be training for a very short time on a very small training set! So the model that is created here will be pretty much worthless. But! These steps will show how Raster Vision experiments are set up and run, so when you are ready to run against a lot of training data for a longer time on a GPU, you'll know what you have to do. Also, we'll show how to make predictions on the data using a model we've already trained on GPUs to show what you can expect to get out of Raster Vision.

For the Quickstart we are going to be using one of the published :ref:`docker containers`
as it has an environment with all necessary dependencies already installed.

.. seealso:: It is also possible to install Raster Vision using ``pip``, but it can be time-consuming and error-prone to install all the necessary dependencies. See :ref:`install raster vision` for more details.

.. note:: This Quickstart requires a Docker installation. We have tested this with Docker 18, although you may be able to use a lower version. See `Get Started with Docker <https://www.docker.com/get-started>`_ for installation instructions.

You'll need to choose two directories, one for keeping your source file and another for
holding experiment output. Make sure these directories exist:

.. code-block:: console

   > export RV_QUICKSTART_CODE_DIR=`pwd`/code
   > export RV_QUICKSTART_EXP_DIR=`pwd`/rv_root
   > mkdir -p ${RV_QUICKSTART_CODE_DIR} ${RV_QUICKSTART_EXP_DIR}

Now we can run a console in the the Docker container by doing

.. code-block:: terminal

   > docker run --rm -it -p 6006:6006 \
        -v ${RV_QUICKSTART_CODE_DIR}:/opt/src/code  \
        -v ${RV_QUICKSTART_EXP_DIR}:/opt/data \
        quay.io/azavea/raster-vision:pytorch-0.11 /bin/bash

.. seealso:: See :ref:`docker containers` for more information about setting up Raster Vision with
             Docker containers.

The Data
--------

.. raw:: html

         <div style="position: relative; padding-bottom: 56.25%; overflow: hidden; max-width: 100%;">
             <iframe src="_static/tiny-spacenet-map.html" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
         </div>

Creating an ExperimentSet
-------------------------

Create a Python file in the ``${RV_QUICKSTART_CODE_DIR}`` named ``tiny_spacenet.py``. Inside, you're going to create an :ref:`experiment set`. You can think of an ``ExperimentSet`` a lot like the ``unittest.TestSuite``: It's a class that contains specially-named methods that are run via reflection by the ``rastervision`` command line tool.

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
           channel_order = [0, 1, 2]
           background_class_id = 2

           # ------------- TASK -------------

           task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                               .with_chip_size(300) \
                               .with_chip_options(chips_per_scene=50) \
                               .with_classes({
                                   'building': (1, 'red'),
                                   'background': (2, 'black')
                               }) \
                               .build()

           # ------------- BACKEND -------------

           backend = rv.BackendConfig.builder(rv.PYTORCH_SEMANTIC_SEGMENTATION) \
               .with_task(task) \
               .with_train_options(
                   batch_size=2,
                   num_epochs=1,
                   debug=True) \
               .build()

           # ------------- TRAINING -------------

           train_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                                                      .with_uri(train_image_uri) \
                                                      .with_channel_order(channel_order) \
                                                      .with_stats_transformer() \
                                                      .build()

           train_label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                                                            .with_vector_source(train_label_uri) \
                                                            .with_rasterizer_options(background_class_id) \
                                                            .build()
           train_label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                                                    .with_raster_source(train_label_raster_source) \
                                                    .build()

           train_scene =  rv.SceneConfig.builder() \
                                        .with_task(task) \
                                        .with_id('train_scene') \
                                        .with_raster_source(train_raster_source) \
                                        .with_label_source(train_label_source) \
                                        .build()

           # ------------- VALIDATION -------------

           val_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                                                    .with_uri(val_image_uri) \
                                                    .with_channel_order(channel_order) \
                                                    .with_stats_transformer() \
                                                    .build()

           val_label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                                                          .with_vector_source(val_label_uri) \
                                                          .with_rasterizer_options(background_class_id) \
                                                          .build()
           val_label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                                                  .with_raster_source(val_label_raster_source) \
                                                  .build()

           val_scene = rv.SceneConfig.builder() \
                                     .with_task(task) \
                                     .with_id('val_scene') \
                                     .with_raster_source(val_raster_source) \
                                     .with_label_source(val_label_source) \
                                     .build()

           # ------------- DATASET -------------

           dataset = rv.DatasetConfig.builder() \
                                     .with_train_scene(train_scene) \
                                     .with_validation_scene(val_scene) \
                                     .build()

           # ------------- EXPERIMENT -------------

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

The ``exp_main`` method has a special name: any method starting with ``exp_`` is one that Raster Vision
will look for experiments in. Raster Vision does this by calling the method and processing any experiments
that are returned - you can either return a single experiment or a list of experiments.

Notice that we set up a ``SceneConfig``, which points to a ``RasterSourceConfig``, and calls
``with_label_source`` with a GeoJSON URI, which sets a default ``LabelSourceConfig`` type into
the scene based on the extension of the URI. We also set a ``StatsTransformer`` to be used
for the ``RasterSource`` by calling ``with_stats_transformer()``,
which sets a default ``StatsTransformerConfig`` onto the ``RasterSourceConfig`` transformers.
This transformer is needed to convert uint16 values in the rasters to the uint8 values needed by the
data loader in PyTorch. (In the future, we plan on relaxing this requirement.)

Running an experiment
---------------------

Now that you've configured an experiment, we can perform a dry run of executing it to see what running the
full workflow will look like:

.. code-block:: console

   > cd /opt/src/code
   > rastervision run local -p tiny_spacenet.py -n

   Ensuring input files exist    [####################################]  100%
   Checking for existing output  [####################################]  100%

   Commands to be run in this order:
   ANALYZE from tiny-spacenet-experiment

   CHIP from tiny-spacenet-experiment
     DEPENDS ON: ANALYZE from tiny-spacenet-experiment

   TRAIN from tiny-spacenet-experiment
     DEPENDS ON: CHIP from tiny-spacenet-experiment

   BUNDLE from tiny-spacenet-experiment
     DEPENDS ON: ANALYZE from tiny-spacenet-experiment
     DEPENDS ON: TRAIN from tiny-spacenet-experiment

   PREDICT from tiny-spacenet-experiment
     DEPENDS ON: ANALYZE from tiny-spacenet-experiment
     DEPENDS ON: TRAIN from tiny-spacenet-experiment

   EVAL from tiny-spacenet-experiment
     DEPENDS ON: ANALYZE from tiny-spacenet-experiment
     DEPENDS ON: PREDICT from tiny-spacenet-experiment

The console output above is what you should expect - although there will be a color scheme
to make things easier to read in terminals that support it.

Here we see that we're about to run the ANALYZE, CHIP, TRAIN, BUNDLE, PREDICT, and EVAL commands,
and what they depend on. You can change the verbosity to get even more dry run output - we won't
list the output here to save space, but give it a try:

.. code-block:: console

   > rastervision -v run local -p tiny_spacenet.py -n
   > rastervision -vv run local -p tiny_spacenet.py -n

When we're ready to run, we just remove the ``-n`` flag:

.. code-block:: console

   > rastervision run local -p tiny_spacenet.py

Seeing Results
---------------

If you go to ``${RV_QUICKSTART_EXP_DIR}`` you should see a folder structure like this.

.. note:: This uses the ``tree`` command which you may need to install first.

.. code-block:: console

   > tree -L 3
    .
    ├── analyze
    │   └── tiny-spacenet-experiment
    │       ├── command-config-0.json
    │       └── stats.json
    ├── bundle
    │   └── tiny-spacenet-experiment
    │       ├── command-config-0.json
    │       └── predict_package.zip
    ├── chip
    │   └── tiny-spacenet-experiment
    │       ├── chips
    │       └── command-config-0.json
    ├── eval
    │   └── tiny-spacenet-experiment
    │       ├── command-config-0.json
    │       └── eval.json
    ├── experiments
    │   └── tiny-spacenet-experiment.json
    ├── predict
    │   └── tiny-spacenet-experiment
    │       ├── command-config-0.json
    │       └── val_scene.tif
    └── train
        └── tiny-spacenet-experiment
            ├── command-config-0.json
            ├── done.txt
            ├── log.csv
            ├── logs
            ├── model
            ├── models
            ├── train-debug-chips.zip
            └── val-debug-chips.zip

Each directory with a command name contains output for that command type across experiments.
The directory inside those have our experiment ID as the name - this is so different experiments
can share ``root_uri``'s without overwriting each other's output. You can also use "keys", e.g.
``.with_chip_key('chip-size-300')`` on an ``ExperimentConfigBuilder`` to set the directory
for a command across experiments, so that they can share command output. This is useful
in the case where many experiments have the same CHIP output, and so you only want to run that
once for many train commands from various experiments. The experiment configuration is also
saved off in the ``experiments`` directory.

Don't get too excited to look at the evaluation results in ``eval/tiny-spacenet-experiment/`` - we
trained a model for 1 step, and the model is likely making random predictions at this point. We would need to
train on a lot more data for a lot longer for the model to become good at this task.

Predict Packages
----------------

To immediately use Raster Vision with a fully trained model, one can make use of the pretrained models in our `Model Zoo <https://github.com/azavea/raster-vision-examples#model-zoo>`_. However, be warned that these models probably won't work well on imagery taken in a different city, with a different ground sampling distance, or different sensor.

For example, to perform semantic segmentation using a MobileNet-based DeepLab model that has been pretrained for Las Vegas, one can type:

.. code-block:: console

   > rastervision predict https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo/vegas-building-seg-pytorch/predict_package.zip https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo/vegas-building-seg/1929.tif prediction.tif
This will perform a prediction on the image ``1929.tif`` using the provided prediction package, and will produce a file called ``predictions.tif`` that contains the predictions.
Notice that the prediction package and the input raster are transparently downloaded via HTTP.
The input image (false color) and predictions are reproduced below.

.. image:: img/vegas/1929.png
  :width: 333
  :alt: The input image

.. image:: img/vegas/predictions.png
  :width: 333
  :alt: The predictions

.. seealso:: You can read more about the :ref:`predict package` concept and the :ref:`predict cli command` CLI command in the documentation.


Next Steps
----------

This is just a quick example of a Raster Vision workflow. For a more complete example of how to train
a model on SpaceNet (optionally using GPUs on AWS Batch), see the SpaceNet examples in the `Raster Vision Examples <https://github.com/azavea/raster-vision-examples>`_ repository.
