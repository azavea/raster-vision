.. _plugins:

Plugins
=======

You can extend Raster Vision easily by writing Plugins. Any ``Config`` that is created
using the :ref:`builder pattern`, that is based on a key (e.g. ``rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION)``) can use plugins.

All of the configurable entities that are constructed like this in the Raster Vision codebase use
the same sort of registration process as Plugins - the difference is that they are registered
internally in the main Raster Vision :ref:`global registry`. Because of this, the best way
to figure out how to build components of Raster Vision that can be plugged in is to study the
codebase.

Creating Plugins
----------------

You'll need to implement an interface for the Plugin, by inhereting from ``Task``, ``Backend``, etc. You will also have to implement a ``Config`` and ``ConfigBuilder`` for your type. The ``Config`` and ``ConfigBuilder`` should likewise inheret from the appropriate parent class - for example, if you are implementing a backend plugin, you'll need to develop implementations of ``Backend``, ``BackendConfig``, and ``BackendConfigBuilder``. The parent class ``__init__`` of ``BackendConfig`` takes a ``backend_type``, which you will have to assign a unique string. This will be the key that
you later refer to in your experiment configurations. For instance, if you developed a new backend that passed in the ``backend_type = "AWESOME"``, you could reference that backend configuration in an experiment like this:

.. code::

   backend = rv.BackendConfig.builder("AWESOME") \
               .with_awesome_property("etc") \
               .build()

You'll need to implement the ``to_proto`` method and the ``Config`` and the ``from_proto`` method on ``ConfigBuilder`` - in the ``.proto`` files for the entity you are creating a plugin for, you'll see a ``google.protobuf.Struct custom_config`` section. This is the field in the protobuf that can handle arbitrary JSON, and should be used in plugins for configuration.

Registering the Plugin
----------------------

Your plugin file or module must define a ``register_plugin`` method with the following signature:

.. code::

   def register_plugin(plugin_registry):
       pass

The ``plugin_regsitry`` that is passed in has a number of methods that allow for registring the plugin with Raster Vision. This is the method that is called on startup of Raster Vision for any plugin configured in the configuration file. See the :ref:`plugin registry api` API reference for more information on registration methods.


Configuring Raster Vision to use your Plugins
-----------------------------------------------

Raster Vision searches for ``register_plugin`` methods in all the files and modules listed in the Raster Vision configuration. See documentation on the :ref:`plugins config section` section of the configuration for more info on how to set this up.

Plugins in remote environments
------------------------------

In order for plugins to work with any :ref:`experiment runner` that executes commands remotely, the
configured files or modules will have to be available to the remote machines. For example, if
you are using AWS Batch, then your plugin cannot be something that is only stored on your local
machine. In that case, you could store the file in S3 or in a repository that the instances
will have access to through HTTP, or you can ensure that the module containing the plugin
is also installed in the remote runner environment (e.g. by baking a Docker container based
on the Raster Vision container that has your plugins installed, and setting up the AWS Batch
job definition to use that container).

Command configurations carry with them the paths and module names of the plugins they use.
This way, the remote environment knows what plugins to load in order to successfully run
the commands.

Example Plugin
--------------

.. click:example::

   # easy_evaluator.py

   from copy import deepcopy

   import rastervision as rv
   from rastervision.evaluation import (Evaluator, EvaluatorConfig,
                                        EvaluatorConfigBuilder)
   from rastervision.protos.evaluator_pb2 import EvaluatorConfig as EvaluatorConfigMsg

   EASY_EVALUATOR = 'EASY_EVALUATOR'


   class EasyEvaluator(Evaluator):
       def __init__(self, message):
           self.message

       def process(self, scenes, tmp_dir):
           print(self.message)


   class EasyEvaluatorConfig(EvaluatorConfig):
       def __init__(self, message):
           super().__init__(EASY_EVALUATOR)

       def to_proto(self):
           msg = EvaluatorConfigMsg(
               evaluator_type=self.evaluator_type, custom_config={ "message": self.message })
           return msg

       def create_evaluator(self):
           return NoopEvaluator(self.message)

       def update_for_command(self, command_type, experiment_config, context=[]):
           return (self, rv.core.CommandIODefinition())


   class NoopEvaluatorConfigBuilder(EvaluatorConfigBuilder):
       def __init__(self, prev=None):
           self.config = {}
           if prev:
               self.config = {
                   'message': prev.message
               }

           super().__init__(EasyEvaluatorConfig, {})

       def from_proto(self, msg):
           return self.with_message(msg.custom_config.get("message"))

       def with_message(self, message):
           b = deepcopy(self)
           b.config['message'] = message
           return b


   def register_plugin(plugin_registry):
       plugin_registry.register_config_builder(rv.EVALUATOR, NOOP_EVALUATOR,
                                               NoopEvaluatorConfigBuilder)


You can set the file location in the path of your Raster Vision plugin configuration in the  ``files``
setting, and then use it in experiments like so (assuming EASY_EVALUATOR was defined the same as above):

.. code::

   evaluator = rv.EvaluatorConfig.builder(EASY_EVALUATOR) \
                                 .with_message("Great job!") \
                                 .build()

You could then set this evaluator on an experiment just as you would an internal evaluator.
