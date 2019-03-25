import unittest
from unittest.mock import Mock

from rastervision.data import (LabelSource, LabelSourceConfig,
                               LabelSourceConfigBuilder,
                               ChipClassificationLabels)
from rastervision.protos.label_source_pb2 \
    import LabelSourceConfig as LabelSourceConfigMsg

from tests.mock import SupressDeepCopyMixin
from tests.mock.raster_source import MOCK_SOURCE


class MockLabelSource(LabelSource):
    def __init__(self):
        self.mock = Mock()

        self.mock.get_labels.return_value = None

    def get_labels(self, window=None):
        result = self.mock.get_labels(window)
        if result is None:
            return ChipClassificationLabels()
        else:
            return result


class MockLabelSourceConfig(SupressDeepCopyMixin, LabelSourceConfig):
    def __init__(self):
        super().__init__(MOCK_SOURCE)
        self.mock = Mock()

        self.mock.to_proto.return_value = None
        self.mock.create_source.return_value = None
        self.mock.update_for_command.return_value = None

    def to_proto(self):
        result = self.mock.to_proto()
        if result is None:
            msg = super().to_proto()
            msg.MergeFrom(LabelSourceConfigMsg(custom_config={}))
            return msg
        else:
            return result

    def create_source(self, task_config, extent, crs_transformer, tmp_dir):
        result = self.mock.create_source(task_config, extent, crs_transformer,
                                         tmp_dir)
        if result is None:
            return MockLabelSource()
        else:
            return result

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        super().update_for_command(command_type, experiment_config, context)
        self.mock.update_for_command(command_type, experiment_config, context)

    def report_io(self, command_type, io_def):
        self.mock.report_io(command_type, io_def)


class MockLabelSourceConfigBuilder(SupressDeepCopyMixin,
                                   LabelSourceConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(MockLabelSourceConfig, {})
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
