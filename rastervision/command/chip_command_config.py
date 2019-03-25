from copy import deepcopy

import rastervision as rv
from rastervision.command import (ChipCommand, CommandConfig,
                                  CommandConfigBuilder, NoOpCommand)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg
from rastervision.rv_config import RVConfig
from rastervision.data import SceneConfig
from rastervision.command.utils import (check_task_type, check_backend_type)
from rastervision.utils.misc import split_into_groups


class ChipCommandConfig(CommandConfig):
    def __init__(self, root_uri, split_id, task, backend, augmentors,
                 train_scenes, val_scenes):
        super().__init__(rv.CHIP, root_uri, split_id)
        self.task = task
        self.backend = backend
        self.augmentors = augmentors
        self.train_scenes = train_scenes
        self.val_scenes = val_scenes

    def create_command(self, tmp_dir=None):
        if len(self.train_scenes) == 0 and len(self.val_scenes) == 0:
            return NoOpCommand()

        if not tmp_dir:
            _tmp_dir = RVConfig.get_tmp_dir()
            tmp_dir = _tmp_dir.name
        else:
            _tmp_dir = tmp_dir

        retval = ChipCommand(self)
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

    def report_io(self):
        io_def = rv.core.CommandIODefinition()
        self.task.report_io(self.command_type, io_def)
        self.backend.report_io(self.command_type, io_def)
        for scene in self.train_scenes:
            scene.report_io(self.command_type, io_def)
        for scene in self.val_scenes:
            scene.report_io(self.command_type, io_def)
        for augmentor in self.augmentors:
            augmentor.report_io(self.command_type, io_def)
        return io_def

    def split(self, num_parts):
        commands = []
        t_scenes = list(map(lambda x: (0, x), self.train_scenes))
        v_scenes = list(map(lambda x: (1, x), self.val_scenes))

        for i, l in enumerate(
                split_into_groups(t_scenes + v_scenes, num_parts)):
            split_t_scenes = list(
                map(lambda x: x[1], filter(lambda x: x[0] == 0, l)))
            split_v_scenes = list(
                map(lambda x: x[1], filter(lambda x: x[0] == 1, l)))
            c = self.to_builder() \
                .with_train_scenes(split_t_scenes) \
                .with_val_scenes(split_v_scenes) \
                .with_split_id(i) \
                .build()
            commands.append(c)
        return commands

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
        check_task_type(self.task)
        if self.backend is None:
            raise rv.ConfigError('Backend not set for ChipCommandConfig. Use '
                                 'with_backend or with_experiment')
        check_backend_type(self.backend)
        if len(self.train_scenes) > 0:
            for s in self.train_scenes:
                if not isinstance(s, SceneConfig):
                    raise rv.ConfigError(
                        'train_scenes must be a list of class SceneConfig, '
                        'got a list of {}'.format(type(s)))
        if len(self.val_scenes) > 0:
            for s in self.val_scenes:
                if not isinstance(s, SceneConfig):
                    raise rv.ConfigError(
                        'val_scenes must be a list of class SceneConfig, '
                        'got a list of {}'.format(type(s)))

    def build(self):
        self.validate()
        return ChipCommandConfig(self.root_uri, self.split_id, self.task,
                                 self.backend, self.augmentors,
                                 self.train_scenes, self.val_scenes)

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
