import rastervision as rv
from rastervision.backend import (Backend, BackendConfig, BackendConfigBuilder)
from rastervision.protos.backend_pb2 import BackendConfig as BackendConfigMsg

from .noop_task import NOOP_TASK

NOOP_BACKEND = 'NOOP_BACKEND'


class NoopBackend(Backend):
    def process_scene_data(self, scene, data, tmp_dir):
        pass

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        pass

    def train(self, tmp_dir):
        pass

    def load_model(self, chips, windows, tmp_dir):
        pass

    def predict(self, chips, windows, tmp_dir):
        pass


class NoopBackendConfig(BackendConfig):
    def __init__(self):
        super().__init__(NOOP_BACKEND)

    def to_proto(self):
        msg = BackendConfigMsg(
            backend_type=self.backend_type, custom_config={})
        return msg

    def create_backend(self, task_config):
        return NoopBackend()

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        return io_def or rv.core.CommandIODefinition()

    def save_bundle_files(self, bundle_dir):
        return (self, [])

    def load_bundle_files(self, bundle_dir):
        return self


class NoopBackendConfigBuilder(BackendConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(NOOP_BACKEND, NoopBackendConfig, {})

    @staticmethod
    def from_proto(msg):
        return NoopBackendConfigBuilder()

    def _applicable_tasks(self):
        return [NOOP_TASK]

    def _process_task(self, task):
        pass


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.BACKEND, NOOP_BACKEND,
                                            NoopBackendConfigBuilder)
