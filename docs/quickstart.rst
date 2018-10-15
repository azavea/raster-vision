.. _quickstart:

Quickstart
==========

For this Quickstart we are going to be using one of the published  :ref:`docker containers`
as it has an environment with all necessary dependencies already installed.

.. seealso:: It is also possible to install Raster Vision using ``pip``, but it can be time-consuming to install all the necessary dependencies. See :ref:`install raster vision` for more details.

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
        quay.io/azavea/raster-vision:cpu-0.8 /bin/bash

.. seealso:: See :ref:`docker containers` for more information about setting up Raster Vision with
             Docker containers.

Creating an ExperimentSet
-------------------------

Create a Python file in the ``${RV_QUICKSTART_CODE_DIR}`` named ``tiny_spacenet.py``. Inside, you're going to create an :ref:`experiment set`. You can think of an ExperimentSet a lot like the ``unittest.TestSuite``: It's a class that contains specially-named methods that are run via reflection by the ``rastervision`` command line tool.

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
                               .with_chip_size(300) \
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
                                     .with_batch_size(1) \
                                     .with_num_steps(2) \
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


The ``exp_main`` method has a special name: any method starting with ``exp_`` is one that Raster Vision
will look for experiments in. Raster Vision does this by calling the method and processing any experiments
that are returned - you can either return a single experiment or a list of experiments.

Notice that we create a ``TaskConfig`` and ``BackendConfig`` that configure Raster Vision to perform
object detection on buildings. In fact, Raster Vision isn't doing any of the heavy lifting of
actually training the model - it's using the
`TensorFlow Object Detection API <https://github.com/tensorflow/models/tree/master/research/object_detection>`_ for that. Raster Vision
just provides a configuration wrapper that sets up all of the options and data for the experiment
workflow that utilizes that library.

You also can see we set up a ``SceneConfig``, which points to a ``RasterSourceConfig``, and calls
``with_label_source`` with a GeoJSON URI, which sets a default ``LabelSourceConfig`` type into
the scene based on the extension of the URI. We also set a ``StatsTransformer`` to be used
for the ``RasterSource`` represented by this configuration by calling ``with_stats_transformer()``,
which sets a default ``StatsTransformerConfig`` onto the ``RasterSourceConfig`` transformers.

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

.. note:: TensorFlow 1.10 will not work on some older computers due to unsupported vector instructions. Consider building a custom wheel to run the newer version of TensorFlow.

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
   │       ├── command-config.json
   │       └── stats.json
   ├── bundle
   │   └── tiny-spacenet-experiment
   │       ├── command-config.json
   │       └── predict_package.zip
   ├── chip
   │   └── tiny-spacenet-experiment
   │       ├── command-config.json
   │       ├── label-map.pbtxt
   │       ├── train-debug-chips.zip
   │       ├── train.record
   │       ├── train_scene-f353604b-7bc6-40b3-b9ce-e6d45cd27e8c.record
   │       ├── val_scene-f3086bc2-6281-4d46-a612-cf04094db1fb.record
   │       ├── validation-debug-chips.zip
   │       └── validation.record
   ├── eval
   │   └── tiny-spacenet-experiment
   │       ├── command-config.json
   │       └── eval.json
   ├── experiments
   │   └── tiny-spacenet-experiment.json
   ├── predict
   │   └── tiny-spacenet-experiment
   │       ├── command-config.json
   │       └── val_scene.json
   └── train
       └── tiny-spacenet-experiment
           ├── checkpoint
           ├── command-config.json
           ├── eval
           ├── model
           ├── model.ckpt.data-00000-of-00001
           ├── model.ckpt.index
           ├── model.ckpt.meta
           ├── pipeline.config
           ├── tiny-spacenet-experiment.tar.gz
           └── train

Each directory with a command name contains output for that command type across experiments.
The directory inside those have our experiment ID as the name - this is so different experiments
can share root_uri's without overwritting each other's output. You can also use "keys", e.g.
``.with_chip_key('chip-size-300')`` on an ``ExperimentConfigBuilder`` to set the directory
for a command across experiments, so that they can share command output. This is useful
in the case where many experiments have the same CHIP output, and so you only want to run that
once for many train commands from various experiments. The experiment configuration is also
saved off in the ``experiments`` directory.

Don't get too excited to look at the evaluation results in ``eval/tiny-spacenet-experiment/`` - we
trained a model for 2 steps, and the model is likely making random predictions at this point. We would need to
train on a lot more data for a lot longer for the model to become good at this task.

Next Steps
----------

This is just a quick example of a Raster Vision workflow. For a more complete example of how to train
a model on SpaceNet (optionally using GPUs on AWS Batch) and view the results in QGIS, see the SpaceNet examples in the `Raster Vision Examples <https://github.com/azavea/raster-vision-examples>`_ repository.
