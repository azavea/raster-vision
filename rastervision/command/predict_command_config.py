from copy import deepcopy

import rastervision as rv
from rastervision.command import (PredictCommand, CommandConfig,
                                  CommandConfigBuilder, NoOpCommand)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg
from rastervision.rv_config import RVConfig
from rastervision.command.utils import (check_backend_type, check_task_type)
from rastervision.utils.misc import split_into_groups


class PredictCommandConfig(CommandConfig):
    def __init__(self, root_uri, split_id, task, backend, scenes):
        super().__init__(rv.PREDICT, root_uri, split_id)
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

        retval = PredictCommand(self)
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

    def report_io(self):
        io_def = rv.core.CommandIODefinition()
        self.task.report_io(self.command_type, io_def)
        self.backend.report_io(self.command_type, io_def)
        for scene in self.scenes:
            scene.report_io(self.command_type, io_def)
        return io_def

    def split(self, num_parts):
        commands = []
        for i, l in enumerate(split_into_groups(self.scenes, num_parts)):
            c = self.to_builder() \
                .with_scenes(l) \
                .with_split_id(i) \
                .build()
            commands.append(c)
        return commands

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
        check_task_type(self.task)

        if self.backend is None:
            raise rv.ConfigError(
                'Backend not set for PredictCommandConfig. Use '
                'with_backend or with_experiment')
        check_backend_type(self.backend)

    def build(self):
        self.validate()
        return PredictCommandConfig(self.root_uri, self.split_id, self.task,
                                    self.backend, self.scenes)

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
