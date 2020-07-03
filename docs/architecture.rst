.. _architecture:

Architecture and Customization
================================

.. _codebase overview:

Codebase Overview
-------------------

The Raster Vision codebase is designed with modularity and flexibility in mind.
There is a main, required package, ``rastervision.pipeline``, which contains functionality for defining and configuring computational pipelines, running them in different environments using parallelism and GPUs, reading and writing to different file systems, and adding and customizing pipelines via a plugin mechanism. In contrast, the "domain logic" of geospatial deep learning using PyTorch, and running on AWS is contained in a set of optional plugin packages. All plugin packages must be under the ``rastervision`` `native namespace package <https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages>`_.

Each of these packages is contained in a separate ``setuptools``/``pip`` package with its own dependencies, including dependencies on other Raster Vision packages. This means that it's possible to install and use subsets of the functionality in Raster Vision. A short summary of the packages is as follows:

* ``rastervision.pipeline``: define and run pipelines
* ``rastervision.aws_s3``: read and write files on S3
* ``rastervision.aws_batch``: run pipelines on Batch
* ``rastervision.core``: chip classification, object detection, and semantic segmentation pipelines that work on geospatial data along with abstractions for running with different :ref:`backends <backend>` and data formats
* ``rastervision.pytorch_learner``: model building and training code using ``torch`` and ``torchvision``, which can be used independently of ``rastervision.core``.
* ``rastervision.pytorch_backend``: adds backends for the pipelines in ``rastervision.core`` using ``rastervision.pytorch_learner`` to do the heavy lifting

The figure below shows the packages, the dependencies between them, and important base classes within each package.

.. image:: img/rv-packages.png
  :alt: The dependencies between Python packages in Raster Vision

.. _pipelines plugins:

Writing pipelines and plugins
-------------------------------------

In this section, we explain the most important aspects of the ``rastervision.pipeline`` package through a series of examples which incrementally build on one another. These examples show how to write custom pipelines and configuration schemas, how to customize an existing pipeline, and how to package the code as a plugin.

The full source code for Examples 1 and 2 is in `rastervision.pipeline_example_plugin1 <https://github.com/azavea/raster-vision/tree/0.12/rastervision_pipeline/rastervision/pipeline_example_plugin1>`_ and Example 3 is in `rastervision.pipeline_example_plugin2 <https://github.com/azavea/raster-vision/tree/0.12/rastervision_pipeline/rastervision/pipeline_example_plugin2>`_ and they can be run from inside the RV Docker image. However, **note that new plugins are typically created in a separate repo and Docker image**, and :ref:`bootstrap` shows how to do this.

.. _example 1:

Example 1: a simple pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Pipeline`` in RV is a class which represents a sequence of commands with a shared configuration in the form of a ``PipelineConfig``. Here is a toy example of these two classes that saves a set of messages to disk, and then prints them all out.

.. literalinclude:: ../rastervision_pipeline/rastervision/pipeline_example_plugin1/sample_pipeline.py
   :language: python
   :caption: rastervision.pipeline_example_plugin1.sample_pipeline

In order to run this, we need a separate Python file with a ``get_config()`` function which provides an instantiation of the ``SamplePipelineConfig``.

.. literalinclude:: ../rastervision_pipeline/rastervision/pipeline_example_plugin1/config1.py
   :language: python
   :caption: rastervision.pipeline_example_plugin1.config1

Finally, in order to package this code as a plugin, and make it usable within the Raster Vision framework, it needs to be in a package directly under the ``rastervision`` `namespace package <https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages>`_, and have a top-level ``__init__.py`` file with a certain structure.

.. literalinclude:: ../rastervision_pipeline/rastervision/pipeline_example_plugin1/__init__.py
   :language: python
   :caption: rastervision.pipeline_example_plugin1.__init__

We can invoke the Raster Vision CLI to run the pipeline using:

.. code-block:: terminal

    > rastervision run inprocess rastervision.pipeline_example_plugin1.config1 -a root_uri /opt/data/pipeline-example/1/ -s 2

    Running save_messages command split 1/2...
    Saved message to /opt/data/pipeline-example/1/alice.txt
    Saved message to /opt/data/pipeline-example/1/bob.txt
    Running save_messages command split 2/2...
    Saved message to /opt/data/pipeline-example/1/susan.txt
    Running print_messages command...
    hello alice!
    hello bob!
    hello susan!

This uses the ``inprocess`` runner, which executes all the commands in a single process locally (which is good for debugging), and uses the ``LocalFileSystem`` to read and write files. The ``-s 2`` option says to use two splits for splittable commands, and the ``-a root_uri /opt/data/sample-pipeline`` option says to pass the ``root_uri`` argument to the ``get_config`` function.

.. _example 2:

Example 2: hierarchical config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example makes some small changes to the previous example, and shows how configurations can be built up hierarchically. However, the main purpose here is to lay the foundation for :ref:`example 3` which shows how to customize the configuration schema and behavior of this pipeline using a plugin. The changes to the previous example are highlighted with comments, but the overall effect  is to delegate making messages to a ``MessageMaker`` class with its own ``MessageMakerConfig`` including a ``greeting`` field.

.. literalinclude:: ../rastervision_pipeline/rastervision/pipeline_example_plugin1/sample_pipeline2.py
   :language: python
   :caption: rastervision.pipeline_example_plugin1.sample_pipeline2

We can configure the pipeline using:

.. literalinclude:: ../rastervision_pipeline/rastervision/pipeline_example_plugin1/config2.py
   :language: python
   :caption: rastervision.pipeline_example_plugin1.config2

The pipeline can then be run with the above configuration using:

.. code-block:: terminal

    > rastervision run inprocess rastervision.pipeline_example_plugin1.config2 -a root_uri /opt/data/pipeline-example/2/ -s 2

    Running save_messages command split 1/2...
    Saved message to /opt/data/pipeline-example/2/alice.txt
    Saved message to /opt/data/pipeline-example/2/bob.txt
    Running save_messages command split 2/2...
    Saved message to /opt/data/pipeline-example/2/susan.txt
    Running print_messages command...
    hola alice!
    hola bob!
    hola susan!

.. _example 3:

Example 3: customizing an existing pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to customize the behavior of an existing pipeline, namely the ``SamplePipeline2`` developed in :ref:`example 2`. That pipeline delegates printing messages to a ``MessageMaker`` class which is configured by ``MessageMakerConfig``. Our goal here is to make it possible to control the number of exclamation points at the end of the message.

By writing a plugin (ie. a plugin to the existing plugin that was developed in the previous two examples), we can add new behavior without modifying any of the original source code from :ref:`example 2`. This mimics the situation plugin writers will be in when they want to modify the behavior of one of the :ref:`geospatial deep learning pipelines <rv pipelines>` without modifying the source code in the main Raster Vision repo.

The code to implement the new configuration and behavior, and a sample configuration are below. (We omit the ``__init__.py`` file since it is similar to the one in the previous plugin.) Note that the new ``DeluxeMessageMakerConfig`` uses inheritance to extend the configuration schema.

.. literalinclude:: ../rastervision_pipeline/rastervision/pipeline_example_plugin2/deluxe_message_maker.py
   :language: python
   :caption: rastervision.pipeline_example_plugin2.deluxe_message_maker

.. literalinclude:: ../rastervision_pipeline/rastervision/pipeline_example_plugin2/config3.py
   :language: python
   :caption: rastervision.pipeline_example_plugin2.config3

We can run the pipeline as follows:

.. code-block:: terminal

    > rastervision run inprocess rastervision.pipeline_example_plugin2.config3 -a root_uri /opt/data/pipeline-example/3/ -s 2

    Running save_messages command split 1/2...
    Saved message to /opt/data/pipeline-example/3/alice.txt
    Saved message to /opt/data/pipeline-example/3/bob.txt
    Running save_messages command split 2/2...
    Saved message to /opt/data/pipeline-example/3/susan.txt
    Running print_messages command...
    hola alice!!!
    hola bob!!!
    hola susan!!!

The output in ``/opt/data/sample-pipeline`` contains a ``pipeline-config.json`` file which is the serialized version of the ``SamplePipeline2Config`` created in ``config3.py``. The serialized configuration is used to transmit the configuration when running a pipeline remotely. It also is a programming language-independent record of the fully-instantiated configuration that was generated by the ``run`` command in conjunction with any command line arguments. Below is the partial contents of this file. The interesting thing to note here is the ``type_hint`` field that appears twice. This is what allows the JSON to be deserialized back into the Python classes that were originally used.(Recall that the ``register_config`` decorator is what tells the ``Registry`` the type hint for each ``Config`` class.)

.. code-block:: json

    {
        "root_uri": "/opt/data/sample-pipeline",
        "type_hint": "sample_pipeline2",
        "names": [
            "alice",
            "bob",
            "susan"
        ],
        "message_uris": [
            "/opt/data/sample-pipeline/alice.txt",
            "/opt/data/sample-pipeline/bob.txt",
            "/opt/data/sample-pipeline/susan.txt"
        ],
        "message_maker": {
            "greeting": "hola",
            "type_hint": "deluxe_message_maker",
            "level": 3
        }
    }

We now have a plugin that customizes an existing pipeline! Being a toy example, this may all seem like overkill. Hopefully, the real power of the ``pipeline`` package becomes more apparent when considering the standard set of plugins distributed with Raster Vision, and how this functionality can be customized with user-created plugins.

.. _customizing rv:

Customizing Raster Vision
--------------------------

When approaching a new problem or dataset with Raster Vision, you may get lucky and be able to apply RV "off-the-shelf". In other cases, Raster Vision can be used after writing scripts to convert data into the appropriate format.

However, sometimes you will need to modify the functionality of RV to suit your problem. In this case, you could modify the RV source code (ie. any of the code in the :ref:`packages <codebase overview>` in the main RV repo). In some cases, this may be necessary, as the right extension points don't exist. In other cases, the functionality may be very widely-applicable, and you would like to :ref:`contribute <contributing>` it to the main repo. Most of the time, however, the functionality will be problem-specific, or is in an embryonic stage of development, and should be implemented in a plugin that resides outside the main repo.

General information about plugins can be found in :ref:`bootstrap` and :ref:`pipelines plugins`. The following are some brief pointers on how to write plugins for different scenarios. In the future, we would like to enhance this section.

* To add commands to an existing ``Pipeline``: write a plugin with subclasses of the ``Pipeline`` and its corresponding ``PipelineConfig`` class. The new ``Pipeline`` should add a method for the new command, and modify the list of ``commands``. Any new configuration should be added to the subclass of the ``PipelineConfig``. Example: running some data pre- or post-processing code in a pipeline.
* To modify commands of an existing ``Pipeline``: same as above except you will override command methods. If a new configuration field is required, you can subclass the ``Config`` class that field resides within. Example: custom chipping functionality for semantic segmentation. You will need to create subclasses of ``SemanticSegmentationChipOptions``, ``SemanticSegmentation``, and ``SemanticSegmentationConfig``.
* To create a substantially new ``Pipeline``: write a new plugin that adds a new ``Pipeline``. See the ``rastervision.core`` plugin, in particular, the contents of the ``rastervision.core.rv_pipeline`` package. If you want to add a new geospatial deep learning pipeline (eg. for chip regression), you may want to override the ``RVPipeline`` class. In other cases that deviate more from ``RVPipeline``, you may want to write a new ``Pipeline`` class with arbitrary commands and logic, but that uses the core model building and training functionality in the ``rastervision.pytorch_learner`` plugin.
* To add the ability to use new file systems or run in new cloud environments: write a plugin that adds a new ``FileSystem`` or ``Runner``. See the ``rastervision.aws_s3`` and ``rastervision.aws_batch`` plugins for examples.
* To use an existing ``RVPipeline`` with a new ``Backend``: write a plugin that adds a subclass of ``Backend`` and ``BackendConfig``. See the ``rastervision.pytorch_backend`` plugin for an example.
* To override model building or training routines in an existing ``PyTorchLearnerBackend``: write a plugin that adds a subclass of ``Learner`` (and ``LearnerConfig``) that overrides ``build_model`` and ``train_step``, and  a subclass of ``PyTorchLearnerBackend`` (and ``PyTorchLearnerBackendConfig``) that overrides the backend so it uses the ``Learner`` subclass.
