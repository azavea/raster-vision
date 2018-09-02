from abc import ABC, abstractmethod

from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg

class CommandConfig(ABC):
    def __init__(self, command_type):
        self.command_type = command_type

    @abstractmethod
    def create_command(self, tmp_dir):
        """Run the command."""
        pass

    def to_proto(self):
        """Returns the protobuf configuration for this config.
        """
        return CommandConfigMsg(command_type = self.command_type)

    @staticmethod
    @abstractmethod
    def builder():
        """Returns a new builder that takes this configuration
           as its starting point.
        """
        pass

    @staticmethod
    def from_proto(msg):
        """Creates a TaskConfig from the specificed protobuf message
        """
        return rv._registry.get_command_config_builder(msg.command_type)() \
                           .from_proto(msg) \
                           .build()

class CommandConfigBuilder(ABC):
    @abstractmethod
    def build(self):
        """Returns the configuration that is built by this builder.
        """
        pass

    @abstractmethod
    def from_proto(self, msg):
        """Return a builder that takes the configuration from the proto message
           as its starting point.
        """
        pass

    @abstractmethod
    def with_experiment(self, experiment):
        """Generate all required information from this experiment.
           It is sufficient to only call 'with_experiment' before
           calling .build()
        """
        pass
