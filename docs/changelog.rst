CHANGELOG
=========

Raster Vision 0.13
-------------------

Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Add support for multiband images `#972 <https://github.com/azavea/raster-vision/pull/972>`_
* Add support for vector output to predict command `#980 <https://github.com/azavea/raster-vision/pull/980>`_
* Add support for weighted loss for classification and semantic segmentation `#977 <https://github.com/azavea/raster-vision/pull/977>`_
* Add multi raster source `#978 <https://github.com/azavea/raster-vision/pull/978>`_
* Add support for fetching and saving external model definitions `#985 <https://github.com/azavea/raster-vision/pull/985>`_
* Add support for external loss definitions `#992 <https://github.com/azavea/raster-vision/pull/992>`_
* Upgrade to pyproj 2.6 `#1000 <https://github.com/azavea/raster-vision/pull/1000>`_
* Add support for arbitrary albumentations transforms `#1001 <https://github.com/azavea/raster-vision/pull/1001>`_
* Minor tweaks to regression learner `#1013 <https://github.com/azavea/raster-vision/pull/1013>`_
* Add ability to specify number of PyTorch reader processes `#1008 <https://github.com/azavea/raster-vision/pull/1008>`_
* Make img_sz specifiable `#1012 <https://github.com/azavea/raster-vision/pull/1012>`_
* Add ignore_last_class capability to segmentation `#1017 <https://github.com/azavea/raster-vision/pull/1017>`_
* Add filtering capability to segmentation sliding window chip generation `#1018 <https://github.com/azavea/raster-vision/pull/1018>`_
* Add raster transformer to remove NaNs from float rasters, add raster transformers to cast to arbitrary numpy types `#1016 <https://github.com/azavea/raster-vision/pull/1016>`_
* Add plot options for regression `#1023 <https://github.com/azavea/raster-vision/pull/1023>`_
* Add ability to use fewer channels w/ pretrained models `#1026 <https://github.com/azavea/raster-vision/pull/1026>`_
* Remove 4GB file size limit from VSI file system, allow streaming reads `#1020 <https://github.com/azavea/raster-vision/pull/1020>`_
* Add reclassification transformer for segmentation label rasters `#1024 <https://github.com/azavea/raster-vision/pull/1024>`_
* Allow filtering out chips based on proportion of NODATA pixels `#1025 <https://github.com/azavea/raster-vision/pull/1025>`_
* Allow ignore_last_class to take either a boolean or the literal 'force'; in the latter case validation of that argument is skipped so that it can be used with external loss functions `#1027 <https://github.com/azavea/raster-vision/pull/1027>`_
* Add ability to crop raster source extent `#1030 <https://github.com/azavea/raster-vision/pull/1030>`_
* Accept immediate geometries in SceneConfig `#1033 <https://github.com/azavea/raster-vision/pull/1033>`_
* Only perform normalization on unsigned integer types `#1028 <https://github.com/azavea/raster-vision/pull/1028>`_
* Make group_uris specifiable and add group_train_sz_rel `#1035 <https://github.com/azavea/raster-vision/pull/1035>`_
* Make number of training and dataloader previews independent of batch size `#1038 <https://github.com/azavea/raster-vision/pull/1038>`_
* Allow continuing training from a model bundle `#1022 <https://github.com/azavea/raster-vision/pull/1022>`_
* Allow reading directly from raster source during training without chipping `#1046 <https://github.com/azavea/raster-vision/pull/1046>`_
* Remove external commands (obsoleted by external architectures and loss functions) `#1047 <https://github.com/azavea/raster-vision/pull/1047>`_
* Allow saving SS predictions as probabilities `#1057 <https://github.com/azavea/raster-vision/pull/1057>`_

Bug Fixes
~~~~~~~~~~~~

* Update all relevant saved URIs in config before instantiating Pipeline `#993 <https://github.com/azavea/raster-vision/pull/993>`_
* Pass verbose flag to batch jobs `#988 <https://github.com/azavea/raster-vision/pull/988>`_
* Fix: Ensure Integer class_id `#990 <https://github.com/azavea/raster-vision/pull/990>`_
* Use ``--ipc=host`` by default when running the docker container `#1077 <https://github.com/azavea/raster-vision/pull/1077>`_

Raster Vision 0.12
-------------------

This release presents a major refactoring of Raster Vision intended to simplify the codebase, and make it more flexible and customizable.

To learn about how to upgrade existing experiment configurations, perhaps the best approach is to read the `source code <https://github.com/azavea/raster-vision/tree/0.12/rastervision_pytorch_backend/rastervision/pytorch_backend/examples>`_ of the :ref:`rv examples` to get a feel for the new syntax. Unfortunately, existing predict packages will not be usable with this release, and upgrading and re-running the experiments will be necessary. For more advanced users who have written plugins or custom commands, the internals have changed substantially, and we recommend reading :ref:`architecture`.

Since the changes in this release are sweeping, it is difficult to enumerate a list of all changes and associated PRs. Therefore, this change log describes the changes at a high level, along with some justifications and pointers to further documentation.

Simplified Configuration Schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We are still using a modular, programmatic approach to configuration, but have switched to using a ``Config`` base class which uses the `Pydantic <https://pydantic-docs.helpmanual.io/>`_ library. This allows us to define configuration schemas in a declarative fashion, and let the underlying library handle serialization, deserialization, and validation. In addition, this has allowed us to `DRY <https://en.wikipedia.org/wiki/Don%27t_repeat_yourself>`_ up the configuration code, eliminate the use of Protobufs, and represent configuration from plugins in the same fashion as built-in functionality. To see the difference, compare the configuration code for ``ChipClassificationLabelSource`` in 0.11 (`label_source.proto <https://github.com/azavea/raster-vision/blob/0.11/rastervision/protos/label_source.proto>`_ and `chip_classification_label_source_config.py <https://github.com/azavea/raster-vision/blob/0.11/rastervision/data/label_source/chip_classification_label_source_config.py>`_), and in 0.12 (`chip_classification_label_source_config.py <https://github.com/azavea/raster-vision/blob/0.12/rastervision_core/rastervision/core/data/label_source/chip_classification_label_source_config.py>`_).

Abstracted out Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~

Raster Vision includes functionality for running computational pipelines in local and remote environments, but previously, this functionality was tightly coupled with the "domain logic" of machine learning on geospatial data in the ``Experiment`` abstraction. This made it more difficult to add and modify commands, as well as use this functionality in other projects. In this release, we factored out the experiment running code into a separate :ref:`rastervision.pipeline <pipelines plugins>` package, which can be used for defining, configuring, customizing, and running arbitrary computational pipelines.

Reorganization into Plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rest of Raster Vision is now written as a set of optional plugins that have  ``Pipelines`` which implement the "domain logic" of machine learning on geospatial data. Implementing everything as optional (``pip`` installable) plugins makes it easier to install subsets of Raster Vision functionality, eliminates separate code paths for built-in and plugin functionality, and provides (de facto) examples of how to write plugins. See :ref:`codebase overview` for more details.

More Flexible PyTorch Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 0.10 release added PyTorch backends for chip classification, semantic segmentation, and object detection. In this release, we abstracted out the common code for training models into a flexible ``Learner`` base class with subclasses for each of the computer vision tasks. This code is in the ``rastervision.pytorch_learner`` plugin, and is used by the ``Backends`` in ``rastervision.pytorch_backend``. By decoupling ``Backends`` and ``Learners``, it is now easier to write arbitrary ``Pipelines`` and new ``Backends`` that reuse the core model training code, which can be customized by overriding methods such as ``build_model``. See :ref:`customizing rv`.

Removed Tensorflow Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tensorflow backends and associated Docker images have been removed. It is too difficult to maintain backends for multiple deep learning frameworks, and PyTorch has worked well for us. Of course, it's still possible to write ``Backend`` plugins using any framework.

Other Changes
~~~~~~~~~~~~~~

* For simplicity, we moved the contents of the `raster-vision-examples <https://github.com/azavea/raster-vision-examples>`_ and `raster-vision-aws <https://github.com/azavea/raster-vision-aws>`_ repos into the main repo. See :ref:`rv examples` and :ref:`cloudformation setup`.
* To help people bootstrap new projects using RV, we added :ref:`bootstrap`.
* All the PyTorch backends now offer data augmentation using `albumentations <https://albumentations.readthedocs.io/>`_.
* We removed the ability to automatically skip running commands that already have output, "tree workflows", and "default providers". We also unified the ``Experiment``, ``Command``, and ``Task`` classes into a single ``Pipeline`` class which is subclassed for different computer vision (or other) tasks. These features and concepts had little utility in our experience, and presented stumbling blocks to outside contributors and plugin writers.
* Although it's still possible to add new ``VectorSources`` and other classes for reading data, our philosophy going forward is to prefer writing pre-processing scripts to get data into the format that Raster Vision can already consume. The ``VectorTileVectorSource`` was removed since it violates this new philosophy.
* We previously attempted to make predictions for semantic segmentation work in a streaming fashion (to avoid running out of RAM), but the implementation was buggy and complex. So we reverted to holding all predictions for a scene in RAM, and now assume that scenes are roughly < 20,000 x 20,000 pixels. This works better anyway from a parallelization standponit.
* We switched to writing chips to disk incrementally during the ``CHIP`` command using a ``SampleWriter`` class to avoid running out of RAM.
* The term "predict package" has been replaced with "model bundle", since it rolls off the tongue better, and ``BUNDLE`` is the name of the command that produces it.
* Class ids are now indexed starting at 0 instead of 1, which seems more intuitive. The "null class", used for marking pixels in semantic segmentation that have not been labeled, used to be 0, and is now equal to ``len(class_ids)``.
* The ``aws_batch`` runner was renamed ``batch`` due to a naming conflict, and the names of the configuration variables for Batch changed. See :ref:`aws batch setup`.

Future Work
~~~~~~~~~~~~

The next big features we plan on developing are:

* the ability to read and write data in `STAC <https://stacspec.org/>`_ format using the `label extension <https://github.com/radiantearth/stac-spec/tree/master/extensions/label>`_. This will facilitate integration with other tools such as `GroundWork <https://groundwork.azavea.com/>`_.

Raster Vision 0.11
-------------------

Features
~~~~~~~~~~

- Added the possibility for chip classification to use data augmentors from the albumentations libary to enhance the training data. `#859 <https://github.com/azavea/raster-vision/pull/859>`_
- Updated the Quickstart doc with pytorch docker image and model `#863 <https://github.com/azavea/raster-vision/pull/863>`_
- Added the possibility to deal with class imbalances through oversampling. `#868 <https://github.com/azavea/raster-vision/pull/868>`_

Raster Vision 0.11.0
~~~~~~~~~~~~~~~~~~~~~

Bug Fixes
^^^^^^^^^^

- Ensure randint args are ints `#849 <https://github.com/azavea/raster-vision/pull/849>`_
- The augmentors were not serialized properly for the chip command  `#857 <https://github.com/azavea/raster-vision/pull/857>`_
- Fix problems with pretrained flag `#860 <https://github.com/azavea/raster-vision/pull/860>`_
- Correctly get_local_path for some zxy tile URIS `#865 <https://github.com/azavea/raster-vision/pull/865>`_

Raster Vision 0.10
------------------

Raster Vision 0.10.0
~~~~~~~~~~~~~~~~~~~~~~

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
* The way to use the ``ConfigBuilders`` for the new backends can be seen in the `examples repo <https://github.com/azavea/raster-vision-examples>`_ and the :ref:`backend` reference

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
