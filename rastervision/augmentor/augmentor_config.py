from abc import abstractmethod

import rastervision as rv
from rastervision.core import (Config, ConfigBuilder)


class AugmentorConfig(Config):
    def __init__(self, augmentor_type):
        self.augmentor_type = augmentor_type

    @abstractmethod
    def create_augmentor(self):
        """Create the Augmentor that this configuration represents"""
        pass

    def to_builder(self, augmentor_type):
        return rv._registry.get_config_builder(rv.AUGMENTOR,
                                               self.augmentor_type)(self)

    @staticmethod
    def builder(augmentor_type):
        return rv._registry.get_config_builder(rv.AUGMENTOR, augmentor_type)()

    @staticmethod
    def from_proto(msg):
        """Creates a AugmentorConfig from the specificed protobuf message
        """
        return rv._registry.get_config_builder(rv.AUGMENTOR, msg.augmentor_type)() \
                           .from_proto(msg) \
                           .build()

    def update_for_command(self, command_type, experiment_config, context=[]):
        # Generally augmentors do not have an affect on the IO.
        return (self, rv.core.CommandIODefinition())


class AugmentorConfigBuilder(ConfigBuilder):
    pass
