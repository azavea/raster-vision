.. _rv2_quickstart:

Quickstart
==========

In this Quickstart, we'll train a semantic segmentation model on `SpaceNet <https://spacenetchallenge.github.io/datasets/datasetHomePage.html>`_ data. Don't get too excited - we'll only be training for a very short time on a very small training set! So the model that is created here will be pretty much worthless. But! These steps will show how Raster Vision pipelines are set up and run, so when you are ready to run against a lot of training data for a longer time on a GPU, you'll know what you have to do. Also, we'll show how to make predictions on the data using a model we've already trained on GPUs to show what you can expect to get out of Raster Vision.

For the Quickstart we are going to be using one of the published :ref:`rv2_docker images`
as it has an environment with all necessary dependencies already installed.

.. seealso:: It is also possible to install Raster Vision using ``pip``, but it can be time-consuming and error-prone to install all the necessary dependencies. See :ref:`rv2_install raster vision` for more details.

.. note:: This Quickstart requires a Docker installation. We have tested this with Docker 18, although you may be able to use a lower version. See `Get Started with Docker <https://www.docker.com/get-started>`_ for installation instructions.

You'll need to choose two directories, one for keeping your configuration source file and another for
holding experiment output. Make sure these directories exist:

.. code-block:: console

   > export RV_QUICKSTART_CODE_DIR=`pwd`/code
   > export RV_QUICKSTART_OUT_DIR=`pwd`/output
   > mkdir -p ${RV_QUICKSTART_CODE_DIR} ${RV_QUICKSTART_OUT_DIR}

Now we can run a console in the the Docker container by doing

.. code-block:: console

   > docker run --rm -it -p 6006:6006 \
        -v ${RV_QUICKSTART_CODE_DIR}:/opt/src/code  \
        -v ${RV_QUICKSTART_OUT_DIR}:/opt/data/output \
        quay.io/azavea/raster-vision:pytorch-0.12 /bin/bash

.. seealso:: See :ref:`rv2_docker images` for more information about setting up Raster Vision with Docker images.

The Data
--------

.. raw:: html

         <div style="position: relative; padding-bottom: 56.25%; overflow: hidden; max-width: 100%;">
             <iframe src="_static/tiny-spacenet-map.html" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
         </div>

Configuring a semantic segmentation pipeline
----------------------------------------------

Create a Python file in the ``${RV_QUICKSTART_CODE_DIR}`` named ``tiny_spacenet.py``. Inside, you're going to write a function called ``get_config`` that returns a ``SemanticSegmentationConfig`` object. This object's type is a subclass of ``PipelineConfig``, and configures a semantic segmentation pipeline which analyzes the imagery, creates training chips, trains a model, makes predictions on validation scenes, evaluates the predictions, and saves a model bundle.

.. code-block:: python

    # tiny_spacenet.py

    from os.path import join

    from rastervision.core.rv_pipeline import *
    from rastervision.core.backend import *
    from rastervision.core.data import *
    from rastervision.pytorch_backend import *
    from rastervision.pytorch_learner import *


    def get_config(runner):
        root_uri = '/opt/data/output/'
        base_uri = ('https://s3.amazonaws.com/azavea-research-public-data/'
                    'raster-vision/examples/spacenet')
        train_image_uri = '{}/RGB-PanSharpen_AOI_2_Vegas_img205.tif'.format(
            base_uri)
        train_label_uri = '{}/buildings_AOI_2_Vegas_img205.geojson'.format(
            base_uri)
        val_image_uri = '{}/RGB-PanSharpen_AOI_2_Vegas_img25.tif'.format(base_uri)
        val_label_uri = '{}/buildings_AOI_2_Vegas_img25.geojson'.format(base_uri)
        channel_order = [0, 1, 2]
        class_config = ClassConfig(
            names=['building', 'background'], colors=['red', 'black'])

        def make_scene(scene_id, image_uri, label_uri):
            """
            - StatsTransformer is used to convert uint16 values to uint8.
            - The GeoJSON does not have a class_id property for each geom,
            so it is inferred as 0 (ie. building) because the default_class_id
            is set to 0.
            - The labels are in the form of GeoJSON which needs to be rasterized
            to use as label for semantic segmentation, so we use a RasterizedSource.
            - The rasterizer set the background (as opposed to foreground) pixels
            to 1 because background_class_id is set to 1.
            """
            raster_source = RasterioSourceConfig(
                uris=[image_uri],
                channel_order=channel_order,
                transformers=[StatsTransformerConfig()])
            label_source = SemanticSegmentationLabelSourceConfig(
                raster_source=RasterizedSourceConfig(
                    vector_source=GeoJSONVectorSourceConfig(
                        uri=label_uri, default_class_id=0, ignore_crs_field=True),
                    rasterizer_config=RasterizerConfig(background_class_id=1)))
            return SceneConfig(
                id=scene_id,
                raster_source=raster_source,
                label_source=label_source)

        dataset = DatasetConfig(
            class_config=class_config,
            train_scenes=[
                make_scene('scene_205', train_image_uri, train_label_uri)
            ],
            validation_scenes=[
                make_scene('scene_25', val_image_uri, val_label_uri)
            ])

        # Use the PyTorch backend for the SemanticSegmentation pipeline.
        chip_sz = 300
        backend = PyTorchSemanticSegmentationConfig(
            model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
            solver=SolverConfig(lr=1e-4, num_epochs=1, batch_sz=2))
        chip_options = SemanticSegmentationChipOptions(
            window_method=SemanticSegmentationWindowMethod.random_sample,
            chips_per_scene=10)

        return SemanticSegmentationConfig(
            root_uri=root_uri,
            dataset=dataset,
            backend=backend,
            train_chip_sz=chip_sz,
            predict_chip_sz=chip_sz,
            chip_options=chip_options)

Running the pipeline
---------------------

We can now run the pipeline by invoking the following command inside the container.

.. code-block:: console

   > export BATCH_CPU_JOB_DEF="" BATCH_CPU_JOB_QUEUE="" BATCH_GPU_JOB_DEF="" BATCH_GPU_JOB_QUEUE="" BATCH_ATTEMPTS="" AWS_S3_REQUESTER_PAYS="False"
   > python -m rastervision.pipeline.cli run inprocess code/tiny_spacenet.py

Seeing Results
---------------

If you go to ``${RV_QUICKSTART_OUT_DIR}`` you should see a directory structure like this.

.. note:: This uses the ``tree`` command which you may need to install first.

.. code-block:: console

   > tree -L 3
    .
    ├── analyze
    │   └── stats.json
    ├── bundle
    │   └── model-bundle.zip
    ├── chip
    │   └── 3113ff8c-5c49-4d3c-8ca3-44d412968108.zip
    ├── eval
    │   └── eval.json
    ├── pipeline-config.json
    ├── predict
    │   └── scene_25.tif
    └── train
        ├── dataloaders
        │   ├── test.png
        │   ├── train.png
        │   └── valid.png
        ├── last-model.pth
        ├── learner-config.json
        ├── log.csv
        ├── model-bundle.zip
        ├── tb-logs
        │   └── events.out.tfevents.1585513048.086fdd4c5530.214.0
        ├── test_metrics.json
        └── test_preds.png

The root directory contains a serialized JSON version of the configuration at ``pipeline-config.json``, and each subdirectory with a command name contains output for that command. You can see test predictions on a batch of data in ``train/test_preds.png``, and evaluation metrics in ``eval/eval.json``, but don't get too excited! We
trained a model for 1 epoch on a tiny dataset, and the model is likely making random predictions at this point. We would need to
train on a lot more data for a lot longer for the model to become good at this task.

Model Bundles
----------------

TODO: update model bundle

To immediately use Raster Vision with a fully trained model, one can make use of the pretrained models in our `Model Zoo <https://github.com/azavea/raster-vision-examples#model-zoo>`_. However, be warned that these models probably won't work well on imagery taken in a different city, with a different ground sampling distance, or different sensor.

For example, to use a Resnet50-DeepLab model that has been trained to do building segmentation on Las Vegas, one can type:

.. code-block:: console

   > rastervision predict https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo/vegas-building-seg-pytorch/predict_package.zip https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo/vegas-building-seg/1929.tif prediction.tif
This will perform a prediction on the image ``1929.tif`` using the provided prediction package, and will produce a file called ``predictions.tif`` that contains the predictions. These files are GeoTiff and you will need a GIS image viewer to open them on your device. Notice that the prediction package and the input raster are transparently downloaded via HTTP.
The input image (false color) and predictions are reproduced below.

.. image:: img/vegas/1929.png
  :width: 333
  :alt: The input image

.. image:: img/vegas/predictions.png
  :width: 333
  :alt: The predictions

.. seealso:: You can read more about the :ref:`rv2_model bundle` concept and the :ref:`rv2_predict cli command` CLI command in the documentation.


Next Steps
----------

This is just a quick example of a Raster Vision pipeline. For a more complete example of how to train
a model on SpaceNet (optionally using GPUs on AWS Batch), see the SpaceNet examples in the `Raster Vision Examples <https://github.com/azavea/raster-vision-examples>`_ repository.
