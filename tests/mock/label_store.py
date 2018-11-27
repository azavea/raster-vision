from unittest.mock import Mock

import rastervision as rv
from rastervision.data import (LabelStore, LabelStoreConfig,
                               LabelStoreConfigBuilder,
                               ChipClassificationLabels)
from rastervision.protos.label_store_pb2 \
    import LabelStoreConfig as LabelStoreConfigMsg

from tests.mock import SupressDeepCopyMixin

MOCK_STORE = 'MOCK_STORE'


class MockLabelStore(LabelStore):
    def __init__(self):
        self.mock = Mock()

        self.mock.get_labels.return_value = None
        self.mock.empty_labels.return_value = None

    def save(self, labels):
        self.mock.save(labels)

    def get_labels(self):
        result = self.mock.get_labels()
        if result is None:
            return ChipClassificationLabels()
        else:
            return result

    def empty_labels(self):
        result = self.mock.empty_labels()
        if result is None:
            return ChipClassificationLabels()
        else:
            return result


class MockLabelStoreConfig(SupressDeepCopyMixin, LabelStoreConfig):
    def __init__(self):
        super().__init__(MOCK_STORE)
        self.mock = Mock()

        self.mock.to_proto.return_value = None
        self.mock.create_store.return_value = None
        self.mock.update_for_command.return_value = None
        self.mock.for_prediction.return_value = None

    def to_proto(self):
        result = self.mock.to_proto()
        if result is None:
            msg = super().to_proto()
            msg.MergeFrom(LabelStoreConfigMsg(custom_config={}))
            return msg
        else:
            return result

    def create_store(self, task_config, extent, crs_transformer, tmp_dir):
        result = self.mock.create_store(task_config, extent, crs_transformer,
                                        tmp_dir)
        if result is None:
            return MockLabelStore()
        else:
            return result

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        result = self.mock.update_for_command(command_type, experiment_config,
                                              context, io_def)
        if result is None:
            return io_def or rv.core.CommandIODefinition()
        else:
            return result

    def for_prediction(self, label_store_uri):
        result = self.mock.for_prediction(label_store_uri)
        if result is None:
            return self
        else:
            return result


class MockLabelStoreConfigBuilder(SupressDeepCopyMixin,
                                  LabelStoreConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(MockLabelStoreConfig, {})
        self.mock = Mock()

        self.mock.from_proto.return_value = None

    def from_proto(self, msg):
        result = self.mock.from_proto(msg)
        if result is None:
            return self
        else:
            return result
