from copy import deepcopy

import rastervision as rv
from rastervision.command import (ChipCommand, CommandConfig,
                                  CommandConfigBuilder, NoOpCommand)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg
from rastervision.rv_config import RVConfig


class ChipCommandConfig(CommandConfig):
    def __init__(self, root_uri, task, backend, augmentors, train_scenes,
                 val_scenes):
        super().__init__(rv.CHIP, root_uri)
        self.task = task
        self.backend = backend
        self.augmentors = augmentors
        self.train_scenes = train_scenes
        self.val_scenes = val_scenes

    def create_command(self, tmp_dir=None):
        if len(self.train_scenes) == 0 and len(self.val_scenes) == 0:
            return NoOpCommand()

        backend = self.backend.create_backend(self.task)
        task = self.task.create_task(backend)

        augmentors = list(map(lambda a: a.create_augmentor(), self.augmentors))

        if not tmp_dir:
            _tmp_dir = RVConfig.get_tmp_dir()
            tmp_dir = _tmp_dir.name
        else:
            _tmp_dir = tmp_dir

        train_scenes = list(
            map(lambda s: s.create_scene(self.task, tmp_dir),
                self.train_scenes))
        val_scenes = list(
            map(lambda s: s.create_scene(self.task, tmp_dir), self.val_scenes))

        retval = ChipCommand(task, augmentors, train_scenes, val_scenes)
        retval.set_tmp_dir(_tmp_dir)
        return retval

    def to_proto(self):
        msg = super().to_proto()

        task = self.task.to_proto()
        backend = self.backend.to_proto()
        train_scenes = list(map(lambda s: s.to_proto(), self.train_scenes))
        val_scenes = list(map(lambda s: s.to_proto(), self.val_scenes))
        msg.MergeFrom(
            CommandConfigMsg(
                chip_config=CommandConfigMsg.ChipConfig(
                    task=task,
                    backend=backend,
                    train_scenes=train_scenes,
                    val_scenes=val_scenes)))

        return msg

    @staticmethod
    def builder():
        return ChipCommandConfigBuilder()


class ChipCommandConfigBuilder(CommandConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(prev)
        if prev is None:
            self.task = None
            self.backend = None
            self.augmentors = []
            self.train_scenes = []
            self.val_scenes = []
        else:
            self.task = prev.task
            self.backend = prev.backend
            self.augmentors = prev.augmentors
            self.train_scenes = prev.train_scenes
            self.val_scenes = prev.val_scenes

    def validate(self):
        super().validate()
        if self.task is None:
            raise rv.ConfigError('Task not set for ChipCommandConfig. Use '
                                 'with_task or with_experiment')
        if self.backend is None:
            raise rv.ConfigError('Backend not set for ChipCommandConfig. Use '
                                 'with_backend or with_experiment')
        if self.train_scenes == []:
            raise rv.ConfigError(
                'Train scenes not set for ChipCommandConfig. Use '
                'with_train_scenes or with_experiment')
        if self.val_scenes == []:
            raise rv.ConfigError(
                'Val scenes not set for ChipCommandConfig. Use '
                'with_val_scenes or with_experiment')

    def build(self):
        self.validate()
        return ChipCommandConfig(self.root_uri, self.task, self.backend,
                                 self.augmentors, self.train_scenes,
                                 self.val_scenes)

    def from_proto(self, msg):
        b = super().from_proto(msg)

        conf = msg.chip_config

        task = rv.TaskConfig.from_proto(conf.task)
        backend = rv.BackendConfig.from_proto(conf.backend)
        augmentors = list(map(rv.AugmentorConfig.from_proto, conf.augmentors))
        train_scenes = list(map(rv.SceneConfig.from_proto, conf.train_scenes))
        val_scenes = list(map(rv.SceneConfig.from_proto, conf.val_scenes))

        b = b.with_task(task)
        b = b.with_backend(backend)
        b = b.with_augmentors(augmentors)
        b = b.with_train_scenes(train_scenes)
        b = b.with_val_scenes(val_scenes)

        return b

    def get_root_uri(self, experiment_config):
        return experiment_config.chip_uri

    def with_experiment(self, experiment_config):
        b = super().with_experiment(experiment_config)
        b = b.with_task(experiment_config.task)
        b = b.with_backend(experiment_config.backend)
        b = b.with_augmentors(experiment_config.dataset.augmentors)
        b = b.with_train_scenes(experiment_config.dataset.train_scenes)
        b = b.with_val_scenes(experiment_config.dataset.validation_scenes)
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

    def with_augmentors(self, augmentors):
        b = deepcopy(self)
        b.augmentors = augmentors
        return b

    def with_train_scenes(self, scenes):
        b = deepcopy(self)
        b.train_scenes = scenes
        return b

    def with_val_scenes(self, scenes):
        b = deepcopy(self)
        b.val_scenes = scenes
        return b
