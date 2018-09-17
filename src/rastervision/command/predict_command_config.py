from copy import deepcopy

import rastervision as rv
from rastervision.command import (PredictCommand, CommandConfig,
                                  CommandConfigBuilder, NoOpCommand)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg


class PredictCommandConfig(CommandConfig):
    def __init__(self, task, backend, scenes):
        super().__init__(rv.PREDICT)
        self.task = task
        self.backend = backend
        self.scenes = scenes

    def create_command(self, tmp_dir):
        if len(self.scenes) == 0:
            return NoOpCommand()

        backend = self.backend.create_backend(self.task)
        task = self.task.create_task(backend)

        scenes = list(
            map(lambda s: s.create_scene(self.task, tmp_dir), self.scenes))

        return PredictCommand(task, scenes)

    def to_proto(self):
        msg = super().to_proto()

        task = self.task.to_proto()
        backend = self.backend.to_proto()
        scenes = list(map(lambda s: s.to_proto(), self.scenes))

        msg.MergeFrom(
            CommandConfigMsg(
                predict_config=CommandConfigMsg.PredictConfig(
                    task=task, backend=backend, scenes=scenes)))

        return msg

    @staticmethod
    def builder():
        return PredictCommandConfigBuilder()


class PredictCommandConfigBuilder(CommandConfigBuilder):
    def __init__(self, prev=None):
        if prev is None:
            self.task = None
            self.backend = None
            self.scenes = []
        else:
            self.task = prev.task
            self.backend = prev.backend
            self.scenes = prev.scenes

    def build(self):
        if self.task is None:
            raise rv.ConfigError(
                'Task not set. Use with_task or with_experiment')

        if self.backend is None:
            raise rv.ConfigError(
                'Backend not set. Use with_backend or with_experiment')

        return PredictCommandConfig(self.task, self.backend, self.scenes)

    def from_proto(self, msg):
        self.process_plugins(msg)
        msg = msg.predict_config

        task = rv.TaskConfig.from_proto(msg.task)
        backend = rv.BackendConfig.from_proto(msg.backend)
        scenes = list(map(rv.SceneConfig.from_proto, msg.scenes))

        b = self.with_task(task)
        b = b.with_backend(backend)
        b = b.with_scenes(scenes)

        return b

    def with_experiment(self, experiment_config):
        b = self.with_task(experiment_config.task)
        b = b.with_backend(experiment_config.backend)
        b = b.with_scenes(experiment_config.dataset.validation_scenes +
                          experiment_config.dataset.test_scenes)
        return b

    def with_task(self, task):
        b = deepcopy(self)
        b.task = task
        return b

    def with_backend(self, backend):
        b = deepcopy(self)
        b.backend = backend
        return b

    def with_scenes(self, scenes):
        b = deepcopy(self)
        b.scenes = scenes
        return b
