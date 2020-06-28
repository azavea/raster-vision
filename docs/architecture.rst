.. _architecture:

Raster Vision Architecture
===========================

.. _codebase overview:

Codebase overview
-------------------

The Raster Vision codebase is designed with modularity and flexibility in mind.
There is a main, required package, ``rastervision.pipeline``, which contains functionality for defining and configuring computational pipelines, running them in different environments using parallelism and GPUs, reading and writing to different file systems, and adding and customizing pipelines via a plugin mechanism. In contrast, the functionality for geospatial data processing, using AWS S3 and Batch, and deep learning using PyTorch is contained in a set of optional packages that contain plugins to the ``rastervision.pipeline`` package. All plugin packages must be under the ``rastervision`` `native namespace package <https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages>`_.

Each of these packages is contained in a separate ``pip`` package with its own dependencies, including dependencies on other Raster Vision packages. This means that it's possible to install and use subsets of the functionality in Raster Vision. A short summary of the packages is as follows:

* ``rastervision.pipeline``: define and run pipelines
* ``rastervision.aws_s3``: read and write files on S3
* ``rastervision.aws_batch``: run pipelines on Batch
* ``rastervision.core``: chip classification, object detection, and semantic segmentation pipelines that work on geospatial data along with abstractions for running with different :ref:`backends <backend>` and data formats
* ``rastervision.pytorch_learner``: model building and training code using ``torch`` and ``torchvision``, which can be used independently of ``rastervision.core``.
* ``rastervision.pytorch_backend``: adds backends for the pipelines in ``rastervision.core`` using ``rastervision.pytorch_learner`` to do the heavy lifting

The figure below shows the packages, the dependencies between them, and important base classes within each package.

.. image:: img/rv-packages.png
  :alt: The dependencies between Python packages in Raster Vision

.. _pipeline package:

The pipeline package
----------------------

In this section, we explain the most important aspects of the ``rastervision.pipeline`` package through a series of examples which incrementally build on one another. The inline comments should be read as an integral part of the documentatation. The code has been lightly edited for brevity, but the full runnable code can be found in ``rastervision.examples``.



.. _example 1:

Example 1: a simple pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Pipeline`` in RV is a class which represents a sequence of commands with a shared configuration in the form of a ``PipelineConfig``. Here is a toy example of these two classes that saves a set of messages to disk, and then prints them all out.

.. code-block:: python

    # Each Config needs to be registered with a type hint which is used for
    # serializing and deserializing to JSON.
    @register_config('sample_pipeline')
    class SamplePipelineConfig(PipelineConfig):
        # Config classes are configuration schemas. Each field is an attributes
        # with a type and optional default value.
        names: List[str] = ['alice', 'bob']
        message_uris: Optional[List[str]] = None

        def build(self, tmp_dir):
            # The build method is used to instantiate the corresponding object
            # using this configuration.
            return SamplePipeline(self, tmp_dir)

        def update(self):
            # The update method is used to set default values as a function of
            # other values.
            if self.message_uris is None:
                self.message_uris = [
                    join(self.root_uri, '{}.txt'.format(name))
                    for name in self.names
                ]

    class SamplePipeline(Pipeline):
        # The order in which commands run. Each command correspond to a method.
        commands: List[str] = ['save_messages', 'print_messages']

        # Split commands can be split up and run in parallel.
        split_commands = ['save_messages']

        # GPU commands are run using GPUs if available. There are no commands worth running
        # on a GPU in this pipeline.
        gpu_commands = []

        def save_messages(self, split_ind=0, num_splits=1):
            # Save a file for each name with a message.

            # The num_splits is the number of parallel jobs to use and
            # split_ind tracks the index of the parallel job. In this case
            # we are splitting on the names/message_uris.
            split_groups = split_into_groups(
                list(zip(self.config.names, self.config.message_uris)), num_splits)
            split_group = split_groups[split_ind]

            for name, message_uri in split_group:
                message = 'hello {}!'.format(name)
                # str_to_file and most functions in the file_system package can
                # read and write transparently to different file systems based on
                # the URI pattern.
                str_to_file(message, message_uri)
                print('Saved message to {}'.format(message_uri))

        def print_messages(self):
            # Read all the message files and print them.
            for message_uri in self.config.message_uris:
                message = file_to_str(message_uri)
                print(message)


In order to run this, we need a separate Python file with a ``get_config()`` function which provides an instantiation of the ``SamplePipelineConfig``.

.. code-block:: python

    def get_config(runner, root_uri):
        # The get_config function returns an instantiated PipelineConfig and
        # plays a similar role as a typical "config file" used in other systems.
        # It's different in that it can have loops, conditionals, local variables,
        # etc. The runner argument is the name of the runner used to run the
        # pipeline (eg. local or batch). Any other arguments are passed from
        # the CLI using the -a option.
        names = ['alice', 'bob', 'susan']

        # Note that root_uri is a field that is inherited from PipelineConfig,
        # the parent class of SamplePipelineConfig, and specifies the root URI
        # where any output files are saved.
        return SamplePipelineConfig(root_uri=root_uri, names=names)

Assuming this config file is at ``my_config.py``, we can invoke the Raster Vision CLI to run the pipeline using

.. code-block:: shell

    > rastervision run inprocess my_config.py -a root_uri /opt/data/sample-pipeline -s 2

    Running save_messages command split 1/2...
    Saved message to /opt/data/sample-pipeline/alice.txt
    Saved message to /opt/data/sample-pipeline/bob.txt
    Running save_messages command split 2/2...
    Saved message to /opt/data/sample-pipeline/susan.txt
    Running print_messages command...
    hello alice!
    hello bob!
    hello susan!

This uses the ``inprocess`` runner, which executes all the commands in a single process locally, and uses the ``LocalFileSystem`` to read and write files. Using the ``aws_batch`` and ``aws_s3`` plugins, it's possible to use the ``batch`` runner to run commands in parallel and using GPUs in the cloud using AWS Batch, and read and write files to AWS S3.

The ``-s 2`` option says to use two splits for splittable commands, and the ``-a root_uri /opt/data/sample-pipeline`` option says to pass the ``root_uri`` argument to the ``get_config`` function.

.. _example 2:

Example 2: hierarchical config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example makes some small changes to the previous example, and shows how configurations can be built up hierarchically. However, the main purpose here is to lay the foundation for :ref:`example 3` which shows how to customize the configuration schema and behavior of this pipeline using a plugin. The changes to the previous example are highlighted with comments, but the overall effect  is to delegate making messages to a ``MessageMaker`` class with its own ``MessageMakerConfig`` including a ``greeting`` field.

.. code-block:: python

    @register_config('message_maker')
    class MessageMakerConfig(Config):
        greeting: str = 'hello'

        def build(self):
            return MessageMaker(self)

    class MessageMaker():
        def __init__(self, config):
            self.config = config

        def make_message(self, name):
            # Use the greeting field to make the message.
            return '{} {}!'.format(self.config.greeting, name)

    @register_config('sample_pipeline2')
    class SamplePipeline2Config(PipelineConfig):
        names: List[str] = ['alice', 'bob']
        message_uris: Optional[List[str]] = None
        # Fields can have other Configs as types.
        message_maker: MessageMakerConfig = MessageMakerConfig()

        def build(self, tmp_dir):
            return SamplePipeline2(self, tmp_dir)

        def update(self):
            if self.message_uris is None:
                self.message_uris = [
                    join(self.root_uri, '{}.txt'.format(name))
                    for name in self.names
                ]

    class SamplePipeline2(Pipeline):
        commands: List[str] = ['save_messages', 'print_messages']
        split_commands = ['save_messages']
        gpu_commands = []

        def save_messages(self, split_ind=0, num_splits=1):
            message_maker = self.config.message_maker.build()

            split_groups = split_into_groups(
                list(zip(self.config.names, self.config.message_uris)), num_splits)
            split_group = split_groups[split_ind]

            for name, message_uri in split_group:
                # Unlike before, we use the message_maker to make the message.
                message = message_maker.make_message(name)
                str_to_file(message, message_uri)
                print('Saved message to {}'.format(message_uri))

        def print_messages(self):
            for message_uri in self.config.message_uris:
                message = file_to_str(message_uri)
                print(message)

We can configure the pipeline in ``my_config.py`` using:

.. code-block:: python

    def get_config(runner, root_uri):
        names = ['alice', 'bob', 'susan']
        # Same as before except we can set the greeting to be
        # 'hola' instead of 'hello'.
        message_maker = MessageMakerConfig(greeting='hola')
        return SamplePipeline2Config(
            root_uri=root_uri, names=names, message_maker=message_maker)

The pipeline can then be run with the above configuration using:

.. code-block:: shell

    > rastervision run inprocess my_config.py -a root_uri /opt/data/sample-pipeline

    Running save_messages command...
    Saved message to /opt/data/sample-pipeline/alice.txt
    Saved message to /opt/data/sample-pipeline/bob.txt
    Saved message to /opt/data/sample-pipeline/susan.txt
    Running print_messages command...
    hola alice!
    hola bob!
    hola susan!

.. _example 3:

Example 3: customizing a pipeline using a plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to customize the behavior of an existing pipeline, namely the ``SamplePipeline2`` developed in :ref:`example 2`. That pipeline delegates printing messages to a ``MessageMaker`` class which is configured by ``MessageMakerConfig``. Our goal here is to make it possible to control the number of exclamation points at the end of the message. This involves modifying both the behavior in ``MessageMaker`` as well as the configuration schema in ``MesageMakerConfig``.


We can implement this as a plugin, which contributes subclasses ``DeluxeMessageMaker`` and ``DeluxeMessageMakerConfig``. By using a plugin, we can add new behavior without modifying any of the original source code from :ref:`example 2`. In order for Raster Vision to discover a plugin, the code must be in a package under the ``rastervision`` `namespace package <https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages>`_. In this case, the package is ``rastervision.deluxe_message_maker``. The other thing needed to define a plugin is for the top-level ``__init__.py`` file to have a particular structure which can be seen below.

.. code-block:: python

    # Code from rastervision.deluxe_message_maker.__init__.py

    # Always need to import this first.
    import rastervision.pipeline

    # Need to import any modules with register_config decorators.
    import rastervision.deluxe_message_maker.deluxe_message_maker

    def register_plugin(registry):
        # Can be used to manually update the registry. Useful
        # for adding new FileSystems and Runners.
        pass

The code to implement the new configuration and behavior, and a sample configuration are below. Note that the new ``Config`` uses inheritance to extend the schema.

.. code-block:: python

    # Code from rastervision.deluxe_message_maker.deluxe_message_maker.py

    # You always need to use the register_config decorator.
    @register_config('deluxe_message_maker')
    class DeluxeMessageMakerConfig(MessageMakerConfig):
        # Note that this inherits the greeting field from MessageMakerConfig.
        level: int = 1

        def build(self):
            return DeluxeMessageMaker(self)

    class DeluxeMessageMaker(MessageMaker):
        def make_message(self, name):
            # Uses the level field to determine the number of exclamation marks.
            exclamation_marks = '!' * self.config.level
            return '{} {}{}'.format(self.config.greeting, name, exclamation_marks)

.. code-block:: python

    # Code from my_config.py

    def get_config(runner, root_uri):
        names = ['alice', 'bob', 'susan']
        # Note that we use the DeluxeMessageMakerConfig and set the level to 3.
        message_maker = DeluxeMessageMakerConfig(greeting='hola', level=3)
        return SamplePipeline2Config(
            root_uri=root_uri, names=names, message_maker=message_maker)

We can run the pipeline as follows:

.. code-block:: shell

    > rastervision run inprocess my_config.py -a root_uri /opt/data/sample-pipeline
    Running save_messages command...
    Saved message to /opt/data/sample-pipeline/alice.txt
    Saved message to /opt/data/sample-pipeline/bob.txt
    Saved message to /opt/data/sample-pipeline/susan.txt
    Running print_messages command...
    hola alice!!!
    hola bob!!!
    hola susan!!!

The output in ``/opt/data/sample-pipeline`` contains a ``pipeline-config.json`` file which is the serialized version of the ``SamplePipeline2Config`` created in ``my_config.py``. The serialized configuration is used to transmit the configuration when running a pipeline remotely. It also is a programming language-independent record of the fully-instantiated configuration that was generated by the ``run`` command in conjunction with any command line arguments. Below is the partial contents of this file. The interesting thing to note here is the ``type_hint`` field that appears twice. This is what allows the JSON to be deserialized back into the Python classes that were originally used.(Recall that the ``register_config`` decorator is what tells the ``Registry`` the type hint for each ``Config`` class.)

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
----------------------------

-use raster vision as is
-add new options
-add new command to existing pipeline
-add new label_source or raster_source
-add new rvpipeline
-add new backend
-add new backend/learner
-add new rvpipeline but use existing backend/learner
