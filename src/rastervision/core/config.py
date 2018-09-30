from abc import (ABC, abstractmethod)
import os

from rastervision.utils.files import download_or_copy


class ConfigError(Exception):
    pass


def set_nested_keys(target,
                    mods,
                    ignore_missing_keys=False,
                    set_missing_keys=False):
    """Sets dictionary keys based on modifications stated
       in a dictionary. TODO: Explain.
       Only overrides values, does not replace dicts.
       TODO: Errors will be user facing, so provide good feedback.
    """
    searched_keys, found_keys = [], []

    def f(_target, _mods):
        for key in _target:
            if key in _mods.keys():
                found_keys.append(key)
                if type(_target[key]) is dict:
                    if type(_mods[key]) is dict:
                        f(_target[key], _mods[key])
                    else:
                        raise ConfigError(
                            'Error: cannot modify dict with value')
                else:
                    _target[key] = _mods[key]
            else:
                if type(_target[key]) is dict:
                    f(_target[key], _mods)
        searched_keys.extend(list(_mods.keys()))

        if set_missing_keys:
            for key in set(_mods.keys()) - set(found_keys):
                if not type(_mods[key]) is dict:
                    _target[key] = _mods[key]
                    found_keys.append(key)

    f(target, mods)
    if not ignore_missing_keys:
        d = set(searched_keys) - set(found_keys)
        if d:
            raise ConfigError(
                'Mod keys not found in target dict: {}'.format(d))


class Config(ABC):
    @abstractmethod
    def to_builder(self):
        """Return a builder based on this config.
        """
        pass

    @abstractmethod
    def to_proto(self):
        """Returns the protobuf configuration for this config.
        """
        pass

    @abstractmethod
    def preprocess_command(self, command_type, experiment_config,
                           context=None):
        """Returns a copy of this config which may or may not have
           been modified based on the command needs and the experiment
           configuration, as well as the IO definitions this configuration
           contributes to the command. [TODO: Reword]

           Args:
              command_type: The command type that is currently being preprocessed.
              experiment_config: The experiment configuration that this configuration
                                 is a part of.
              context: Optional list of parent configurations, to allow
                       for child configurations contained in collections
                       to understand their context in the experiment configuration.

           Note: While configuration aims to be immutable for client
                 facing operations, this is an internal operation and
                 mutating the coniguration is acceptable.

           Returns: (config, io_def)
        """
        pass

    @staticmethod
    @abstractmethod
    def builder():
        """Returns a new builder that takes this configuration
           as its starting point.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_proto(msg):
        """Creates a Config from the specificed protobuf message
        TODO: Allow loading from file uri or dict
        """
        pass


class ConfigBuilder(ABC):
    def __init__(self, config_class, config=None):
        """Construct a builder.

           Args:
             config_class: The Config class that this builder builds.
             config: A dictionary of **kwargs that will eventually be passed
                     into the __init__ method of config_class to build the configuration.
                     This config is modified with the fluent builder methods.
        """
        if config is None:
            config = {}

        self.config_class = config_class
        self.config = config

    def build(self):
        """Returns the configuration that is built by this builder.
        """
        self.validate()
        return self.config_class(**self.config)

    def validate(self):
        """Validate this config, if there is validation on the builder that
           is not captured by the required arguments of the config.
        """
        pass

    @abstractmethod
    def from_proto(self, msg):
        """Return a builder that takes the configuration from the proto message
           as its starting point.
        """
        pass


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
        pass

    def bundle_file(self, uri, bundle_dir):
        local_path = download_or_copy(uri, bundle_dir)
        base_name = os.path.basename(local_path)
        return (local_path, base_name)

    @abstractmethod
    def load_bundle_files(self, bundle_dir):
        """Load files from a prediction package bundle directory."""
        pass
