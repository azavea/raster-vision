from copy import deepcopy

import rastervision as rv
from rastervision.command import (PredictCommand, CommandConfig,
                                  CommandConfigBuilder, NoOpCommand)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg
from rastervision.rv_config import RVConfig


class PredictCommandConfig(CommandConfig):
    def __init__(self, root_uri, task, backend, scenes):
        super().__init__(rv.PREDICT, root_uri)
        self.task = task
        self.backend = backend
        self.scenes = scenes

    def create_command(self, tmp_dir=None):
        if len(self.scenes) == 0:
            return NoOpCommand()

        if not tmp_dir:
            _tmp_dir = RVConfig.get_tmp_dir()
            tmp_dir = _tmp_dir.name
        else:
            _tmp_dir = tmp_dir

        backend = self.backend.create_backend(self.task)
        task = self.task.create_task(backend)

        scenes = list(
            map(lambda s: s.create_scene(self.task, tmp_dir), self.scenes))
        retval = PredictCommand(task, scenes)
        retval.set_tmp_dir(_tmp_dir)
        return retval

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
        super().__init__(prev)
        if prev is None:
            self.task = None
            self.backend = None
            self.scenes = []
        else:
            self.task = prev.task
            self.backend = prev.backend
            self.scenes = prev.scenes

    def validate(self):
        super().validate()
        if self.task is None:
            raise rv.ConfigError('Task not set for PredictCommandConfig. Use '
                                 'with_task or with_experiment')

        if self.backend is None:
            raise rv.ConfigError(
                'Backend not set for PredictCommandConfig. Use '
                'with_backend or with_experiment')

    def build(self):
        self.validate()
        return PredictCommandConfig(self.root_uri, self.task, self.backend,
                                    self.scenes)

    def from_proto(self, msg):
        b = super().from_proto(msg)

        conf = msg.predict_config

        task = rv.TaskConfig.from_proto(conf.task)
        backend = rv.BackendConfig.from_proto(conf.backend)
        scenes = list(map(rv.SceneConfig.from_proto, conf.scenes))

        b = b.with_task(task)
        b = b.with_backend(backend)
        b = b.with_scenes(scenes)

        return b

    def get_root_uri(self, experiment_config):
        return experiment_config.predict_uri

    def with_experiment(self, experiment_config):
        b = super().with_experiment(experiment_config)
        b = b.with_task(experiment_config.task)
        b = b.with_backend(experiment_config.backend)
        b = b.with_scenes(experiment_config.dataset.validation_scenes +
                          experiment_config.dataset.test_scenes)
        return b

    def with_task(self, task):
        """Sets a specific task type.

        Args:
            task:  A TaskConfig object.

        """
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
