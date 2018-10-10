from abc import abstractmethod
from copy import deepcopy

import rastervision as rv
from rastervision.rv_config import RVConfig
from rastervision.core import (Config, ConfigBuilder, BundledConfigMixin,
                               CommandIODefinition)


class BackendConfig(BundledConfigMixin, Config):
    def __init__(self, backend_type, pretrained_model_uri=None):
        self.backend_type = backend_type
        self.pretrained_model_uri = pretrained_model_uri

    @abstractmethod
    def create_backend(self, task_config):
        """Create the Backend that this configuration represents

           Args:
              task_config: The task configuration for the task
                           to be accomplished by this backend.
        """
        pass

    def to_builder(self):
        return rv._registry.get_config_builder(rv.BACKEND,
                                               self.backend_type)(self)

    @staticmethod
    def builder(backend_type):
        return rv._registry.get_config_builder(rv.BACKEND, backend_type)()

    @staticmethod
    def from_proto(msg):
        """Creates a BackendConfig from the specificed protobuf message
        """
        return rv._registry.get_config_builder(rv.BACKEND, msg.backend_type)() \
                           .from_proto(msg) \
                           .build()

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        io_def = CommandIODefinition()
        if command_type == rv.TRAIN:
            if self.pretrained_model_uri:
                io_def.add_input(self.pretrained_model_uri)
        return (self, io_def)


class BackendConfigBuilder(ConfigBuilder):
    def __init__(self, backend_type, config_class, config=None, prev=None):
        if config is None:
            config = {}
        if prev:
            config['train_options'] = prev.train_options
        super().__init__(config_class, config)
        self.task = None
        self.backend_type = backend_type

    @abstractmethod
    def _applicable_tasks(self):
        """Returns the tasks that this backend can be applied to.
        """
        pass

    @abstractmethod
    def _process_task(self, task):
        """Subclasses override this to set up configuration related
           to this task
        """
        pass

    def from_proto(self, msg):
        return self.with_pretrained_model(msg.pretrained_model_uri)

    def with_task(self, task):
        """Sets a specific task type.

        Args:
            task:  A TaskConfig object.

        """
        if task.task_type not in self._applicable_tasks():
            raise Exception(
                'Backend of type {} cannot be applied to task type {}'.format(
                    task.task_type, self.backend_type))
        b = deepcopy(self)
        b.task = task
        b = b._process_task()
        return b

    def with_pretrained_model(self, uri):
        """Set a pretrained model URI. The filetype and meaning
           for this model will be different based on the backend implementation.
        """
        b = deepcopy(self)
        b.config['pretrained_model_uri'] = uri
        return b

    def with_model_defaults(self, model_defaults_key):
        """Sets the backend configuration and pretrained model defaults
           according to the model defaults configuration.
        """
        model_defaults = RVConfig.get_instance().get_model_defaults()

        if self.backend_type in model_defaults:
            backend_defaults = model_defaults[self.backend_type]
            if model_defaults_key in backend_defaults:
                return self._load_model_defaults(
                    backend_defaults[model_defaults_key])
            else:
                raise rv.ConfigError('No defaults found for model key {}'
                                     .format(model_defaults_key))
        else:
            raise rv.ConfigError('No model defaults for backend {}'
                                 .format(self.backend_type))
        return self

    def _load_model_defaults(self, model_defaults):
        """Overriding classes should handle this if they
           want to allow default parameters to be loaded
           from the default configurations.
        """
        return self
