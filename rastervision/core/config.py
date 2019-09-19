from abc import (ABC, abstractmethod)
import os
import inspect

from rastervision.utils.files import download_or_copy


class ConfigError(Exception):
    pass


class Config(ABC):
    @abstractmethod
    def to_builder(self):
        """Return a builder based on this config.
        """
        pass  # pragma: no cover

    @abstractmethod
    def to_proto(self):
        """Returns the protobuf configuration for this config.
        """
        pass  # pragma: no cover

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        """Updates this configuration for the given command

        Note: While configuration is immutable for client facing operations,
        this is an internal operation and mutates the configuration.

        Args:
            command_type: The command type that is currently being
                preprocessed. experiment_config: The experiment configuration
                that this configuration is a part of.
            context: Optional list of parent configurations, to allow for child
                configurations contained in collections to understand their
                context in the experiment configuration.

        Returns:
          Nothing. Call should mutate the configuration object itself.
        """
        pass  # pragma: no cover

    @abstractmethod
    def report_io(self, command_type, io_def):
        """Updates the given CommandIODefinition.

        So that it includes the inputs, outputs, and missing files for this
        configuration at this command.

        Args:
            command_type: The command type that is currently being preprocessed.
            io_def: The CommandIODefinition that this call should modify.

        Returns: Nothing. This call should make the appropriate calls to the
            given io_def to mutate its state.
        """
        pass

    @staticmethod
    @abstractmethod
    def builder():
        """Returns a new builder that takes this configuration
           as its starting point.
        """
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def from_proto(msg):
        """Creates a Config from the specificed protobuf message
        TODO: Allow loading from file uri or dict
        """
        pass  # pragma: no cover


class ConfigBuilder(ABC):
    def __init__(self, config_class, config=None):
        """Construct a builder.

           Args:
             config_class: The Config class that this builder builds.
             config: A dictionary of **kwargs that will eventually be passed
                     into the __init__ method of config_class to build the configuration.
                     This config is modified with the fluent builder methods.
        """
        if config is None:  # pragma: no cover
            config = {}

        self.config_class = config_class
        self.config = config

    def build(self):
        """Returns the configuration that is built by this builder.
        """
        self.validate()
        arguments = set(inspect.getfullargspec(self.config_class).args)
        keys = set(self.config.keys())
        config = {k: self.config[k] for k in (arguments & keys)}
        return self.config_class(**config)

    def validate(self):
        """Validate this config, if there is validation on the builder that
           is not captured by the required arguments of the config.
        """
        pass  # pragma: no cover

    @abstractmethod
    def from_proto(self, msg):
        """Return a builder that takes the configuration from the proto message
           as its starting point.
        """
        pass  # pragma: no cover


class BundledConfigMixin(ABC):
    """Mixin for configurations that participate in the bundling of a
    prediction package"""
    @abstractmethod
    def save_bundle_files(self, bundle_dir):
        """Place files into a bundle directory for bundling into
        a prediction package.

        Returns: A tuple of (config, uris) of the modified configuration
                 with the basenames of URIs in place of the original URIs,
                 and a list of URIs that are to be bundled.
        """
        pass  # pragma: no cover

    def bundle_file(self, uri, bundle_dir):
        local_path = download_or_copy(uri, bundle_dir)
        base_name = os.path.basename(local_path)
        return (local_path, base_name)

    @abstractmethod
    def load_bundle_files(self, bundle_dir):
        """Load files from a prediction package bundle directory."""
        pass  # pragma: no cover
