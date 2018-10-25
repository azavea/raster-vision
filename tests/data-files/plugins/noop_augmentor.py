import rastervision as rv
from rastervision.augmentor import (Augmentor, AugmentorConfig,
                                    AugmentorConfigBuilder)
from rastervision.protos.augmentor_pb2 import AugmentorConfig as AugmentorConfigMsg

from .noop_utils import noop

NOOP_AUGMENTOR = 'NOOP_AUGMENTOR'


class NoopAugmentor(Augmentor):
    def process(self, training_data, tmp_dir):
        return noop(training_data)


class NoopAugmentorConfig(AugmentorConfig):
    def __init__(self):
        super().__init__(NOOP_AUGMENTOR)

    def to_proto(self):
        msg = AugmentorConfigMsg(
            augmentor_type=self.augmentor_type, custom_config={})
        return msg

    def create_augmentor(self):
        return NoopAugmentor()

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        return io_def or rv.core.CommandIODefinition()


class NoopAugmentorConfigBuilder(AugmentorConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(NoopAugmentorConfig, {})

    def from_proto(self, msg):
        return self


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.AUGMENTOR, NOOP_AUGMENTOR,
                                            NoopAugmentorConfigBuilder)
