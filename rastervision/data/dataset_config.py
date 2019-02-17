import logging

import click

import rastervision as rv
from rastervision.augmentor import AugmentorConfig
from rastervision.data import (SceneConfig, Dataset)
from rastervision.core.config import (Config, ConfigBuilder)
from rastervision.protos.dataset_pb2 import DatasetConfig as DatasetConfigMsg
from rastervision.cli import Verbosity

log = logging.getLogger(__name__)


class DatasetConfig(Config):
    def __init__(self,
                 train_scenes=None,
                 validation_scenes=None,
                 test_scenes=None,
                 augmentors=None):
        if train_scenes is None:
            train_scenes = []
        if validation_scenes is None:
            validation_scenes = []
        if test_scenes is None:
            test_scenes = []
        if augmentors is None:
            augmentors = []

        self.train_scenes = train_scenes
        self.validation_scenes = validation_scenes
        self.test_scenes = test_scenes
        self.augmentors = augmentors

    def all_scenes(self):
        return self.train_scenes + \
            self.validation_scenes + \
            self.test_scenes

    def to_builder(self):
        return DatasetConfigBuilder(self)

    def create_dataset(self,
                       task_config,
                       tmp_dir,
                       include_train=True,
                       include_val=True,
                       include_test=True):
        train_scenes = []
        if include_train:
            train_scenes = list(
                map(lambda x: x.create_scene(task_config, tmp_dir),
                    self.train_scenes))

        val_scenes = []
        if include_val:
            val_scenes = list(
                map(lambda x: x.create_scene(task_config, tmp_dir),
                    self.validation_scenes))

        test_scenes = []
        if include_test:
            test_scenes = list(
                map(lambda x: x.create_scene(task_config, tmp_dir),
                    self.test_scenes))

        augmentors = list(map(lambda x: x.create_augmentor(), self.augmentors))

        return Dataset(
            train_scenes=train_scenes,
            validation_scenes=val_scenes,
            test_scenes=test_scenes,
            augmentors=augmentors)

    def to_proto(self):
        """Returns the protobuf configuration for this config.
        """
        train_scenes = list(map(lambda x: x.to_proto(), self.train_scenes))
        val_scenes = list(map(lambda x: x.to_proto(), self.validation_scenes))
        test_scenes = list(map(lambda x: x.to_proto(), self.test_scenes))

        augmentors = list(map(lambda x: x.to_proto(), self.augmentors))

        return DatasetConfigMsg(
            train_scenes=train_scenes,
            validation_scenes=val_scenes,
            test_scenes=test_scenes,
            augmentors=augmentors)

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        verbosity = Verbosity.get()

        def update_scenes(scenes_to_update, ensure_label_store):
            for scene in scenes_to_update:
                if ensure_label_store:
                    # Ensure there is a label store associated with
                    # validation and test scenes on PREDICT command.
                    if not scene.label_store:
                        scene.label_store = scene.to_builder() \
                                                 .with_task(experiment_config.task) \
                                                 .with_label_store() \
                                                 .build() \
                                                 .label_store
                scene.update_for_command(command_type, experiment_config,
                                         context)

        if command_type in [rv.ANALYZE, rv.CHIP]:
            log.debug(
                'Updating train scenes for command {}'.format(command_type))
            if verbosity >= Verbosity.VERBOSE:
                with click.progressbar(
                        self.train_scenes,
                        label='Updating train scenes') as scenes_to_update:
                    update_scenes(scenes_to_update, False)
            else:
                update_scenes(self.train_scenes, False)

        if command_type in [
                rv.ANALYZE, rv.CHIP, rv.PREDICT, rv.EVAL, rv.BUNDLE
        ]:
            log.debug('Updating validation scenes for command {}'.format(
                command_type))
            if Verbosity.get() >= Verbosity.VERBOSE:
                with click.progressbar(
                        self.validation_scenes,
                        label='Updating validation scenes...  '
                ) as scenes_to_update:
                    update_scenes(scenes_to_update, command_type == rv.PREDICT)
            else:
                update_scenes(self.validation_scenes,
                              command_type == rv.PREDICT)

        if command_type in [rv.ANALYZE, rv.PREDICT, rv.EVAL, rv.BUNDLE]:

            log.debug(
                'Updating test scenes for command {}'.format(command_type))
            if Verbosity.get() >= Verbosity.VERBOSE:
                with click.progressbar(
                        self.test_scenes,
                        label='Updating test scenes...  ') as scenes_to_update:
                    update_scenes(scenes_to_update, command_type == rv.PREDICT)
            else:
                update_scenes(self.test_scenes, command_type == rv.PREDICT)

        if command_type == rv.CHIP:
            log.debug(
                'Updating augmentors for command {}'.format(command_type))
            for augmentor in self.augmentors:
                augmentor.update_for_command(command_type, experiment_config,
                                             context)

    def report_io(self, command_type, io_def):
        if command_type in [rv.ANALYZE, rv.CHIP]:
            for scene in self.train_scenes:
                scene.report_io(command_type, io_def)

        if command_type in [
                rv.ANALYZE, rv.CHIP, rv.PREDICT, rv.EVAL, rv.BUNDLE
        ]:
            for scene in self.validation_scenes:
                scene.report_io(command_type, io_def)

        if command_type in [rv.ANALYZE, rv.PREDICT, rv.EVAL, rv.BUNDLE]:
            for scene in self.test_scenes:
                scene.report_io(command_type, io_def)

        if command_type == rv.CHIP:
            for augmentor in self.augmentors:
                augmentor.report_io(command_type, io_def)

        return io_def

    @staticmethod
    def from_proto(msg):
        """Creates a TaskConfig from the specificed protobuf message
        """
        return DatasetConfigBuilder().from_proto(msg).build()

    @staticmethod
    def builder():
        return DatasetConfigBuilder()


class DatasetConfigBuilder(ConfigBuilder):
    def __init__(self, prev=None):
        config = {
            'train_scenes': [],
            'validation_scenes': [],
            'test_scenes': [],
            'augmentors': []
        }
        if prev:
            config['train_scenes'] = prev.train_scenes
            config['validation_scenes'] = prev.validation_scenes
            config['test_scenes'] = prev.test_scenes
            config['augmentors'] = prev.augmentors
        super().__init__(DatasetConfig, config)

    def _copy(self):
        """Create a copy; avoid using deepcopy as it can have
        performance implications with many scenes
        """
        ds = DatasetConfigBuilder()
        ds.config['train_scenes'] = self.config['train_scenes']
        ds.config['validation_scenes'] = self.config['validation_scenes']
        ds.config['test_scenes'] = self.config['test_scenes']
        ds.config['augmentors'] = self.config['augmentors']
        return ds

    def from_proto(self, msg):
        train_scenes = list(
            map(lambda x: SceneConfig.from_proto(x), msg.train_scenes))
        val_scenes = list(
            map(lambda x: SceneConfig.from_proto(x), msg.validation_scenes))
        test_scenes = list(
            map(lambda x: SceneConfig.from_proto(x), msg.test_scenes))
        augmentors = list(
            map(lambda x: AugmentorConfig.from_proto(x), msg.augmentors))
        return self \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(val_scenes) \
            .with_test_scenes(test_scenes) \
            .with_augmentors(augmentors)

    def with_train_scenes(self, scenes):
        """Sets the scenes to be used for training."""
        b = self._copy()
        b.config['train_scenes'] = list(scenes)
        return b

    def with_train_scene(self, scene):
        """Sets the scene to be used for training."""
        return self.with_train_scenes([scene])

    def with_validation_scenes(self, scenes):
        """Sets the scenes to be used for validation."""
        b = self._copy()
        b.config['validation_scenes'] = list(scenes)
        return b

    def with_validation_scene(self, scene):
        """Sets the scene to be used for validation."""
        return self.with_validation_scenes([scene])

    def with_test_scenes(self, scenes):
        """Sets the scenes to be used for testing."""
        b = self._copy()
        b.config['test_scenes'] = list(scenes)
        return b

    def with_test_scene(self, scene):
        """Sets the scene to be used for testing."""
        return self.with_test_scenes([scene])

    def with_augmentors(self, augmentors):
        """Sets the data augmentors to be used."""
        b = self._copy()
        b.config['augmentors'] = list(augmentors)
        return b

    def with_augmentor(self, augmentor):
        """Sets the data augmentor to be used."""
        return self.with_augmentors([augmentor])
