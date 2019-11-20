CHANGELOG
=========

Raster Vision 0.11
-------------------

Features
~~~~~~~~~~

- Added the possibility for chip classification to use data augmentors from the albumentations libary to enhance the training data. `#859 <https://github.com/azavea/raster-vision/pull/859>`__
- For chip classification: multi-GPU systems are supported. By default all GPU's will be used for training and prediction. No user input required. `#861 <https://github.com/azavea/raster-vision/pull/861>`__

Raster Vision 0.11.0
~~~~~~~~~~~~~~~~~~~~~

Bug Fixes
^^^^^^^^^^

- Ensure randint args are ints `#849 <https://github.com/azavea/raster-vision/pull/849>`__
- The augmentors were not serialized properly for the chip command  `#857 <https://github.com/azavea/raster-vision/pull/857>`__
- Fix problems with pretrained flag `#860 <https://github.com/azavea/raster-vision/pull/860>`__

Raster Vision 0.10
------------------

Raster Vision 0.10.0
~~~~~~~~~~~~~~~~~~~

Notes on switching to PyTorch-based backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current backends based on Tensorflow have several problems:

* They depend on third party libraries (Deeplab, TF Object Detection API) that are complex, not well suited to being used as dependencies within a larger project, and are each written in a different style. This makes the code for each backend very different from one other, and unnecessarily complex. This increases the maintenance burden, makes it difficult to customize, and makes it more difficult to implement a consistent set of functionality between the backends.
* Tensorflow, in the maintainer's opinion, is more difficult to write and debug than PyTorch (although this is starting to improve).
* The third party libraries assume that training images are stored as PNG or JPG files. This limits our ability to handle more than three bands and more that 8-bits per channel. We have recently completed some research on how to train models on > 3 bands, and we plan on adding this functionality to Raster Vision.

Therefore, we are in the process of sunsetting the Tensorflow backends (which will probably be removed) and have implemented replacement PyTorch-based backends. The main things to be aware of in upgrading to this version of Raster Vision are as follows:

* Instead of there being CPU and GPU Docker images (based on Tensorflow), there are now tf-cpu, tf-gpu, and pytorch (which works on both CPU and GPU) images. Using ``./docker/build --tf`` or ``./docker/build --pytorch`` will only build the TF or PyTorch images, respectively.
* Using the TF backends requires being in the TF container, and similar for PyTorch. There are now ``--tf-cpu``, ``--tf-gpu``, and ``--pytorch-gpu`` options for the ``./docker/run`` command. The default setting is to use the PyTorch image in the standard (CPU) Docker runtime.
* The `raster-vision-aws <https://github.com/azavea/raster-vision-aws>`_ CloudFormation setup creates Batch resources for TF-CPU, TF-GPU, and PyTorch. It also now uses default AMIs provided by AWS, simplifying the setup process.
* To easily switch between running TF and PyTorch jobs on Batch, we recommend creating two separate Raster Vision profiles with the Batch resources for each of them.
* The way to use the ``ConfigBuilders`` for the new backends can be seen in the `examples repo <https://github.com/azavea/raster-vision-examples>`_ and the :ref:`backend api reference`

Features
^^^^^^^^^^^^

- Add confusion matrix as metric for semantic segmentation `#788 <https://github.com/azavea/raster-vision/pull/788>`__
- Add predict_chip_size as option for semantic segmentation `#786 <https://github.com/azavea/raster-vision/pull/786>`__
- Handle "ignore" class for semantic segmentation `#783 <https://github.com/azavea/raster-vision/pull/783>`__
- Add stochastic gradient descent ("SGD") as an optimizer option for chip classification `#792 <https://github.com/azavea/raster-vision/pull/792>`__
- Add option to determine if all touched pixels should be rasterized for rasterized RasterSource `#803 <https://github.com/azavea/raster-vision/pull/803>`_
- Script to generate GeoTIFF from ZXY tile server `#811 <https://github.com/azavea/raster-vision/pull/811>`_
- Remove QGIS plugin `#818 <https://github.com/azavea/raster-vision/pull/818>`_
- Add PyTorch backends and add PyTorch Docker image `#821 <https://github.com/azavea/raster-vision/pull/821>`_ and `#823 <https://github.com/azavea/raster-vision/pull/823>`_.

Bug Fixes
^^^^^^^^^

- Fixed issue with configuration not being able to read lists `#784 <https://github.com/azavea/raster-vision/pull/784>`__
- Fixed ConfigBuilders not supporting type annotations in __init__ `#800 <https://github.com/azavea/raster-vision/pull/800>`__

Raster Vision 0.9
-----------------

Raster Vision 0.9.0
~~~~~~~~~~~~~~~~~~~

Features
^^^^^^^^
- Add requester_pays RV config option `#762 <https://github.com/azavea/raster-vision/pull/762>`_
- Unify Docker scripts `#743 <https://github.com/azavea/raster-vision/pull/743>`_
- Switch default branch to master `#726 <https://github.com/azavea/raster-vision/pull/726>`_
- Merge GeoTiffSource and ImageSource into RasterioSource `#723 <https://github.com/azavea/raster-vision/pull/723>`_
- Simplify/clarify/test/validate RasterSource `#721 <https://github.com/azavea/raster-vision/pull/721>`_
- Simplify and generalize geom processing `#711 <https://github.com/azavea/raster-vision/pull/711>`_
- Predict zero for nodata pixels on semantic segmentation `#701 <https://github.com/azavea/raster-vision/pull/701>`_
- Add support for evaluating vector output with AOIs `#698 <https://github.com/azavea/raster-vision/pull/698>`_
- Conserve disk space when dealing with raster files `#692 <https://github.com/azavea/raster-vision/pull/692>`_
- Optimize StatsAnalyzer `#690 <https://github.com/azavea/raster-vision/pull/690>`_
- Include per-scene eval metrics `#641 <https://github.com/azavea/raster-vision/pull/641>`_
- Make and save predictions and do eval chip-by-chip `#635 <https://github.com/azavea/raster-vision/pull/635>`_
- Decrease semseg memory usage `#630 <https://github.com/azavea/raster-vision/pull/630>`_
- Add support for vector tiles in .mbtiles files `#601 <https://github.com/azavea/raster-vision/pull/601>`_
- Add support for getting labels from zxy vector tiles `#532 <https://github.com/azavea/raster-vision/pull/532>`_
- Remove custom ``__deepcopy__`` implementation from ``ConfigBuilder``\s. `#567 <https://github.com/azavea/raster-vision/pull/567>`_
- Add ability to shift raster images by given numbers of meters. `#573 <https://github.com/azavea/raster-vision/pull/573>`_
- Add ability to generate GeoJSON segmentation predictions. `#575 <https://github.com/azavea/raster-vision/pull/575>`_
- Add ability to run the DeepLab eval script.  `#653 <https://github.com/azavea/raster-vision/pull/653>`_
- Submit CPU-only stages to a CPU queue on Aws.  `#668 <https://github.com/azavea/raster-vision/pull/668>`_
- Parallelize CHIP and PREDICT commands  `#671 <https://github.com/azavea/raster-vision/pull/671>`_
- Refactor ``update_for_command`` to split out the IO reporting into ``report_io``. `#671 <https://github.com/azavea/raster-vision/pull/671>`_
- Add Multi-GPU Support to DeepLab Backend `#590 <https://github.com/azavea/raster-vision/pull/590>`_
- Handle multiple AOI URIs `#617 <https://github.com/azavea/raster-vision/pull/617>`_
- Give ``train_restart_dir`` Default Value `#626 <https://github.com/azavea/raster-vision/pull/626>`_
- Use ```make`` to manage local execution `#664 <https://github.com/azavea/raster-vision/pull/664>`_
- Optimize vector tile processing  `#676 <https://github.com/azavea/raster-vision/pull/676>`_

Bug Fixes
^^^^^^^^^
- Fix Deeplab resume bug: update path in checkpoint file `#756 <https://github.com/azavea/raster-vision/pull/756>`_
- Allow Spaces in ``--channel-order`` Argument `#731 <https://github.com/azavea/raster-vision/pull/731>`_
- Fix error when using predict packages with AOIs `#674 <https://github.com/azavea/raster-vision/pull/674>`_
- Correct checkpoint name `#624 <https://github.com/azavea/raster-vision/pull/624>`_
- Allow using default stride for semseg sliding window  `#745 <https://github.com/azavea/raster-vision/pull/745>`_
- Fix filter_by_aoi for ObjectDetectionLabels `#746 <https://github.com/azavea/raster-vision/pull/746>`_
- Load null channel_order correctly `#733 <https://github.com/azavea/raster-vision/pull/733>`_
- Handle Rasterio crs that doesn't contain EPSG `#725 <https://github.com/azavea/raster-vision/pull/725>`_
- Fixed issue with saving semseg predictions for non-georeferenced imagery `#708 <https://github.com/azavea/raster-vision/pull/708>`_
- Fixed issue with handling width > height in semseg eval `#627 <https://github.com/azavea/raster-vision/pull/627>`_
- Fixed issue with experiment configs not setting key names correctly `#576 <https://github.com/azavea/raster-vision/pull/576>`_
- Fixed issue with Raster Sources that have channel order `#576 <https://github.com/azavea/raster-vision/pull/576>`_


Raster Vision 0.8
-----------------

Raster Vision 0.8.1
~~~~~~~~~~~~~~~~~~~

Bug Fixes
^^^^^^^^^
- Allow multiploygon for chip classification `#523 <https://github.com/azavea/raster-vision/pull/523>`_
- Remove unused args for AWS Batch runner `#503 <https://github.com/azavea/raster-vision/pull/503>`_
- Skip over lines when doing chip classification, Use background_class_id for scenes with no polygons `#507 <https://github.com/azavea/raster-vision/pull/507>`_
- Fix issue where ``get_matching_s3_keys`` fails when ``suffix`` is ``None`` `#497 <https://github.com/azavea/raster-vision/pull/497>`_
