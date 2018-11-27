from unittest.mock import Mock

from rastervision.task import (Task, TaskConfig, TaskConfigBuilder)
from rastervision.protos.task_pb2 import TaskConfig as TaskConfigMsg

from tests.mock import SupressDeepCopyMixin

MOCK_TASK = 'MOCK_TASK'


class MockTask(Task):
    def __init__(self, task_config, backend):
        self.config = task_config
        self.backend = backend
        self.mock = Mock()

        self.mock.get_train_windows.return_value = None
        self.mock.get_train_labels.return_value = None
        self.mock.get_predict_windows.return_value = None

    def get_train_windows(self, scene):
        result = self.mock.get_train_windows(scene)
        if result is None:
            return []
        else:
            return result

    def get_train_labels(self, window, scene):
        result = self.mock.get_train_labels(window, scene)
        if result is None:
            return []
        else:
            return result

    def post_process_predictions(self, labels, scene):
        return self.mock.post_process_predictions(labels, scene)

    def get_predict_windows(self, extent):
        result = self.mock.get_predict_windows(extent)
        if result is None:
            return []
        else:
            return result

    def save_debug_predict_image(self, scene, debug_dir_uri):
        return self.mock.save_debug_predict_image(scene, debug_dir_uri)


class MockTaskConfig(SupressDeepCopyMixin, TaskConfig):
    def __init__(self):
        super().__init__(MOCK_TASK)
        self.mock = Mock()

        self.mock.create_task.return_value = None
        self.mock.to_proto.return_value = None
        self.mock.update_for_command.return_value = None
        self.mock.save_bundle_files.return_value = (self, [])
        self.mock.load_bundle_files.return_value = self

    def create_task(self, backend):
        result = self.mock.create_task(backend)
        if result is None:
            return MockTask(self, backend)
        else:
            return result

    def to_proto(self):
        result = self.mock.to_proto()
        if result is None:
            return TaskConfigMsg(task_type=self.task_type, custom_config={})
        else:
            return result

    def save_bundle_files(self, bundle_dir):
        return self.mock.save_bundle_files(bundle_dir)

    def load_bundle_files(self, bundle_dir):
        return self.mock.load_bundle_files(bundle_dir)

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        result = self.mock.update_for_command(command_type, experiment_config,
                                              context, io_def)
        if result is None:
            # Have input always be this file, and output be a non-existant file,
            # so commands always run

            io_def = super().update_for_command(
                command_type, experiment_config, context, io_def)
            io_def.add_input(__file__)
            io_def.add_output('{}{}'.format(__file__, 'xxxx'))
            return io_def
        else:
            return result


class MockTaskConfigBuilder(SupressDeepCopyMixin, TaskConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(MockTaskConfig, {})
        self.mock = Mock()

        self.mock.from_proto.return_value = None

    def from_proto(self, msg):
        result = self.mock.from_proto(msg)
        if result is None:
            return MockTaskConfigBuilder()
        else:
            return result
