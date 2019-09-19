from abc import ABC, abstractmethod
from copy import deepcopy

import rastervision as rv
from rastervision.plugin import PluginRegistry
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg


class CommandConfig(ABC):
    def __init__(self, command_type, root_uri, split_id=0):
        self.command_type = command_type
        self.root_uri = root_uri
        self.split_id = split_id

    @abstractmethod
    def create_command(self):
        """Run the command."""
        pass

    @abstractmethod
    def report_io(self):
        """Returns an IODefinition object with all inputs and outputs reported
        for this command.
        """
        pass

    def split(self, num_parts):
        """Split this command config into num_parts parts if possible.
        This will return a list of size between 1 and num_parts containing command
        configurations. If the command configuration does not split, will return [self]
        """
        return [self]

    def utilizes_gpu(self):
        """Method that determines if this command can utilize a GPU.
        This is useful for runners to know what resources to utilize
        for command execution - e.g. don't spin up the more expensive
        GPU machines if this command can't use the GPUs.

        Defaults to False.
        """
        return False

    def to_proto(self):
        """Returns the protobuf configuration for this config.
        """
        plugin_config = PluginRegistry.get_instance().to_proto()
        return CommandConfigMsg(command_type=self.command_type,
                                root_uri=self.root_uri,
                                split_id=self.split_id,
                                plugins=plugin_config)

    def to_builder(self):
        return rv._registry.get_command_config_builder(self.command_type)(self)

    @staticmethod
    def builder(command_type):
        """Returns a new builder for the given command type.
        """
        return rv._registry.get_command_config_builder(command_type)()

    @staticmethod
    def from_proto(msg):
        """Creates a TaskConfig from the specificed protobuf message
        """
        command_type = msg.command_type
        return rv._registry.get_command_config_builder(command_type)() \
                           .from_proto(msg) \
                           .build()


class CommandConfigBuilder(ABC):
    def __init__(self, command_type, prev):
        self.command_type = command_type

        if prev is None:
            self.root_uri = None
            self.split_id = 0
        else:
            self.root_uri = prev.root_uri
            self.split_id = prev.split_id

    @abstractmethod
    def build(self, prev=None):
        """Returns the configuration that is built by this builder.
        """
        pass

    @abstractmethod
    def get_root_uri(self, experiment_config):
        """Return the root URI for this command for a given experiment"""
        pass

    def from_proto(self, msg):
        """Return a builder that takes the configuration from the proto message
           as its starting point.
        """

        # Process plugins from a command config protobuf message.
        if msg.HasField('plugins'):
            PluginRegistry.get_instance().add_plugins_from_proto(msg.plugins)

        return self.with_root_uri(msg.root_uri).with_split_id(msg.split_id)

    def validate(self):
        if self.root_uri is None:
            raise rv.ConfigError(
                'root_uri not set. Use with_root_uri or with_experiment')

    def with_experiment(self, experiment_config):
        """Generate all required information from this experiment.
           It is sufficient to only call 'with_experiment' before
           calling .build()
        """
        return self.with_root_uri(self.get_root_uri(experiment_config))

    def with_root_uri(self, root_uri):
        b = deepcopy(self)
        b.root_uri = root_uri
        return b

    def with_split_id(self, split_id):
        b = deepcopy(self)
        b.split_id = split_id
        return b
