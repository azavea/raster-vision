Usage overview
==============

.. toctree::
    :maxdepth: 2
    :hidden:

    Overview <self>
    basics
    tutorials/index

Users can use Raster Vision in a couple of different ways depending on their needs and level of experience:

* :ref:`As a library <_usage_library>` of utilities for handling geospatial data and training deep learning models that you can incorporate into your own code.
* :ref:`As a low-code framework <_usage_framework>` in the form of the :doc:`Raster Vision Pipeline <../framework/index>` that internally handles all aspects of the training workflow for you and only requires you to configure a few parameters.

As a library
------------

This allows you to pick and choose which parts of Raster Vision to use. For example, to apply your existing PyTorch training code to geospatial imagery, simply create a :ref:`usage_geodataset` like so:

.. code-block:: python

    from rastervision.pytorch_learner import SemanticSegmentationSlidingWindowGeoDataset

    ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
        class_config=class_config,
        image_uri=image_uri,
        label_raster_uri=label_uri,
        size=200,
        stride=100)


As a framework
--------------

This allows you to configure a full pipeline in one go like so:

.. literalinclude:: /../rastervision_pytorch_backend/rastervision/pytorch_backend/examples/tiny_spacenet.py
    :language: python
    :pyobject: get_config

And then run it from the command line:

.. code-block:: console

    > rastervision run local <path/to/file.py>

Read more about it here: :doc:`../framework/index`.
