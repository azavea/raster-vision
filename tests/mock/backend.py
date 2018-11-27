from unittest.mock import Mock

import rastervision as rv
from rastervision.backend import (Backend, BackendConfig, BackendConfigBuilder)
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg

from tests.mock import SupressDeepCopyMixin
from .task import MOCK_TASK

MOCK_BACKEND = 'MOCK_BACKEND'


class MockBackend(Backend):
    def __init__(self):
        self.mock = Mock()

        self.mock.predict.return_value = None

    def process_scene_data(self, scene, data, tmp_dir):
        return self.mock.process_scene_data(scene, data, tmp_dir)

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        return self.mock.process_sceneset_results(training_results,
                                                  validation_results, tmp_dir)

    def train(self, tmp_dir):
        return self.mock.train(tmp_dir)

    def load_model(self, tmp_dir):
        return self.mock.load_model(tmp_dir)

    def predict(self, chips, windows, tmp_dir):
        result = self.mock.predict(chips, windows, tmp_dir)
        if result is None:
            return rv.data.ChipClassificationLabels()
        else:
            return result


class MockBackendConfig(SupressDeepCopyMixin, BackendConfig):
    def __init__(self):
        super().__init__(MOCK_BACKEND)
        self.mock = Mock()

        self.mock.to_proto.return_value = None
        self.mock.create_backend.return_value = None
        self.mock.update_for_command.return_value = None
        self.mock.save_bundle_files.return_value = (self, [])
        self.mock.load_bundle_files.return_value = self

    def to_proto(self):
        result = self.mock.to_proto()
        if result is None:
            return BackendConfigMsg(
                backend_type=self.backend_type, custom_config={})
        else:
            return result

    def create_backend(self, task_config):
        result = self.mock.create_backend(task_config)
        if result is None:
            return MockBackend()
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

    def save_bundle_files(self, bundle_dir):
        return self.mock.save_bundle_files(bundle_dir)

    def load_bundle_files(self, bundle_dir):
        return self.mock.load_bundle_files(bundle_dir)


class MockBackendConfigBuilder(SupressDeepCopyMixin, BackendConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(MOCK_BACKEND, MockBackendConfig, {})
        self.mock = Mock()

        self.mock.from_proto.return_value = None
        self.mock._applicable_tasks.return_value = None

    def from_proto(self, msg):
        result = self.mock.from_proto(msg)
        if result is None:
            return MockBackendConfigBuilder()
        else:
            return result

    def _applicable_tasks(self):
        result = self.mock._applicable_tasks
        if result is None:
            return [MOCK_TASK]
        else:
            return result

    def _process_task(self, task):
        self.mock._process_task(task)
