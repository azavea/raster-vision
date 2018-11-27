import rastervision as rv
from rastervision.task import (Task, TaskConfig, TaskConfigBuilder)
from rastervision.protos.task_pb2 import TaskConfig as TaskConfigMsg

NOOP_TASK = 'NOOP_TASK'


class NoopTask(Task):
    def get_train_windows(self, scene):
        return []

    def get_train_labels(self, window, scene):
        return []

    def post_process_predictions(self, labels):
        return labels

    def get_predict_windows(self, extent):
        return []

    def save_debug_predict_image(self, scene, debug_dir_uri):
        pass

    def make_chips(self, train_scenes, validation_scenes, augmentors, tmp_dir):
        pass

    def train(self, tmp_dir):
        pass

    def predict(self, scenes, tmp_dir):
        pass


class NoopTaskConfig(TaskConfig):
    def __init__(self):
        super().__init__(NOOP_TASK)

    def to_proto(self):
        msg = TaskConfigMsg(task_type=self.task_type, custom_config={})
        return msg

    def create_task(self, backend):
        return NoopTask(self, backend)

    def save_bundle_files(self, bundle_dir):
        return (self, [])

    def load_bundle_files(self, bundle_dir):
        return self


class NoopTaskConfigBuilder(TaskConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(NoopTaskConfig, {})

    def from_proto(self, msg):
        return self


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.TASK, NOOP_TASK,
                                            NoopTaskConfigBuilder)
