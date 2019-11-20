import unittest
from unittest.mock import Mock

from rastervision.augmentor import (Augmentor, AugmentorConfig,
                                    AugmentorConfigBuilder)
from rastervision.protos.augmentor_pb2 import AugmentorConfig as AugmentorConfigMsg

from tests.mock import SupressDeepCopyMixin

MOCK_AUGMENTOR = 'MOCK_AUGMENTOR'


class MockAugmentor(Augmentor):
    def __init__(self):
        self.mock = Mock()

        self.mock.process.return_value = None

    def process(self, training_data, tmp_dir):
        result = self.mock.process(training_data, tmp_dir)
        if result is None:
            return training_data
        else:
            return result


class MockAugmentorConfig(SupressDeepCopyMixin, AugmentorConfig):
    def __init__(self):
        super().__init__(MOCK_AUGMENTOR)
        self.mock = Mock()

        self.mock.to_proto.return_value = None
        self.mock.create_augmentor.return_value = None
        self.mock.update_for_command.return_value = None

    def to_proto(self):
        result = self.mock.to_proto()
        if result is None:
            return AugmentorConfigMsg(
                augmentor_type=self.augmentor_type, custom_config={})
        else:
            return result

    def create_augmentor(self):
        result = self.mock.create_augmentor()
        if result is None:
            return MockAugmentor()
        else:
            return result

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        super().update_for_command(command_type, experiment_config, context)
        self.mock.update_for_command(command_type, experiment_config, context)

    def report_io(self, command_type, io_def):
        self.mock.report_io(command_type, io_def)


class MockAugmentorConfigBuilder(SupressDeepCopyMixin, AugmentorConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(MockAugmentorConfig, {})
        self.mock = Mock()

        self.mock.from_proto.return_value = None

    def from_proto(self, msg):
        result = self.mock.from_proto(msg)
        if result is None:
            return self
        else:
            return result


if __name__ == '__main__':
    unittest.main()
