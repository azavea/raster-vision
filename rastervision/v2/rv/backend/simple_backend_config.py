from os.path import join
from copy import deepcopy
from abc import abstractmethod
import json

from google.protobuf import struct_pb2

import rastervision as rv
from rastervision.v2.backend import (BackendConfig, BackendConfigBuilder)
from rastervision.v2.protos.backend_pb2 import BackendConfig as BackendConfigMsg


class BackendOptions():
    """Options that pertain to backends created using SimpleBackendConfig."""

    def __init__(self,
                 chip_uri=None,
                 train_uri=None,
                 train_done_uri=None,
                 model_uri=None,
                 pretrained_uri=None):
        self.chip_uri = chip_uri
        self.train_uri = train_uri
        self.train_done_uri = train_done_uri
        self.model_uri = model_uri
        self.pretrained_uri = pretrained_uri


class SimpleBackendConfig(BackendConfig):
    """A simplified BackendConfig interface.

    This class can be subclassed to created BackendConfigs with less effort
    and a small loss of flexibility when compared to directly subclassing
    BackendConfig. See subclasses of this for examples of how to write your
    own subclass.
    """

    def __init__(self, backend_opts, train_opts):
        """Constructor.

        Args:
            backend_opts: (BackendOptions)
            train_opts: (train_opts_class) object containing options that are
                set by with_train_options()
        """
        super().__init__(self.backend_type)
        self.backend_opts = backend_opts
        self.train_opts = train_opts

    @property
    @abstractmethod
    def train_opts_class(self):
        """The class that holds options set by with_train_options in the builder."""
        pass

    @property
    @abstractmethod
    def backend_type(self):
        """The string representing this backend used in the registry."""
        pass

    @property
    @abstractmethod
    def backend_class(self):
        """The class of the actual Backend that this class configures."""
        pass

    def to_proto(self):
        config = {}
        for k, v in self.backend_opts.__dict__.items():
            config[k] = v
        for k, v in self.train_opts.__dict__.items():
            config[k] = v

        custom_config = struct_pb2.Struct()
        custom_config['json'] = json.dumps(config)
        msg = BackendConfigMsg(
            backend_type=self.backend_type, custom_config=custom_config)
        return msg

    def create_backend(self, task_config):
        return self.backend_class(task_config, self.backend_opts,
                                  self.train_opts)

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        super().update_for_command(command_type, experiment_config, context)

        if command_type == rv.CHIP:
            self.backend_opts.chip_uri = join(experiment_config.chip_uri,
                                              'chips')
        elif command_type == rv.TRAIN:
            self.backend_opts.train_uri = experiment_config.train_uri
            self.backend_opts.model_uri = join(experiment_config.train_uri,
                                               'model')
            self.backend_opts.train_done_uri = join(
                experiment_config.train_uri, 'done.txt')

    def report_io(self, command_type, io_def):
        super().report_io(command_type, io_def)

        if command_type == rv.CHIP:
            io_def.add_output(self.backend_opts.chip_uri)
        elif command_type == rv.TRAIN:
            io_def.add_input(self.backend_opts.chip_uri)
            io_def.add_output(self.backend_opts.model_uri)
            io_def.add_output(self.backend_opts.train_done_uri)
        elif command_type in [rv.PREDICT, rv.BUNDLE]:
            if not self.backend_opts.model_uri:
                io_def.add_missing('Missing model_uri.')
            else:
                io_def.add_input(self.backend_opts.model_uri)
                io_def.add_input(self.backend_opts.train_done_uri)
        elif command_type == rv.EVAL:
            io_def.add_input()

    def save_bundle_files(self, bundle_dir):
        model_uri = self.backend_opts.model_uri
        if not model_uri:
            raise rv.ConfigError('model_uri is not set.')
        local_path, base_name = self.bundle_file(model_uri, bundle_dir)
        new_config = self.to_builder() \
                         .with_model_uri(base_name) \
                         .build()
        return (new_config, [local_path])

    def load_bundle_files(self, bundle_dir):
        model_uri = self.backend_opts.model_uri
        if not model_uri:
            raise rv.ConfigError('model_uri is not set.')
        local_model_uri = join(bundle_dir, model_uri)
        return self.to_builder() \
                   .with_model_uri(local_model_uri) \
                   .build()


class SimpleBackendConfigBuilder(BackendConfigBuilder):
    """A simplified BackendConfigBuilder interface.

    This class can be subclassed to created BackendConfigBuilders with less
    effort and a small loss of flexibility when compared to directly
    subclassing BackendConfigBuilder. See subclasses of this for examples of
    how to write your own subclass.
    """

    def __init__(self, prev_config=None):
        self.backend_opts = BackendOptions()
        self.train_opts = self.config_class.train_opts_class()

        if prev_config:
            self.backend_opts = prev_config.backend_opts
            self.train_opts = prev_config.train_opts

        super().__init__(self.config_class.backend_type, self.config_class)
        self.require_task = prev_config is None

    @property
    @abstractmethod
    def config_class(self):
        """The corresponding Config class for this builder."""
        pass

    @abstractmethod
    def with_train_options(self):
        """Sets the training options which are passed to the Backend."""
        pass

    def build(self):
        self.validate()
        return self.config_class(self.backend_opts, self.train_opts)

    def from_proto(self, msg):
        b = super().from_proto(msg)
        custom_config = msg.custom_config

        if 'json' in custom_config:
            config = json.loads(custom_config['json'])
        else:
            config = custom_config

        for k in self.backend_opts.__dict__.keys():
            if k in config:
                setattr(b.backend_opts, k, config[k])
        for k in self.train_opts.__dict__.keys():
            if k in config:
                setattr(b.train_opts, k, config[k])
        b.require_task = None
        return b

    def validate(self):
        super().validate()

        if self.require_task and not self.task:
            raise rv.ConfigError('You must specify the task this backend '
                                 "is for - use 'with_task'.")

        return True

    def _process_task(self):
        return self

    def with_model_uri(self, model_uri):
        b = deepcopy(self)
        b.backend_opts.model_uri = model_uri
        return b

    def with_pretrained_uri(self, pretrained_uri):
        b = deepcopy(self)
        b.backend_opts.pretrained_uri = pretrained_uri
        return b
