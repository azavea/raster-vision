from copy import deepcopy

import rastervision as rv
from rastervision.augmentor import AugmentorConfig
from rastervision.data import (SceneConfig, Dataset)
from rastervision.core.config import (Config, ConfigBuilder)
from rastervision.protos.dataset_pb2 import DatasetConfig as DatasetConfigMsg


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

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        io_def = rv.core.CommandIODefinition()

        if command_type in [rv.ANALYZE, rv.CHIP]:
            train_scenes = []
            for scene in self.train_scenes:
                (new_config, scene_io_def) = scene.update_for_command(
                    command_type, experiment_config, context)
                io_def.merge(scene_io_def)
                train_scenes.append(new_config)
        else:
            train_scenes = self.train_scenes

        if command_type in [
                rv.ANALYZE, rv.CHIP, rv.PREDICT, rv.EVAL, rv.BUNDLE
        ]:
            val_scenes = []
            for scene in self.validation_scenes:
                if command_type == rv.PREDICT:
                    # Ensure there is a label store associated with
                    # predict and validation scenes.
                    if not scene.label_store:
                        scene = scene.to_builder() \
                                     .with_task(experiment_config.task) \
                                     .with_label_store() \
                                     .build()
                (new_config, scene_io_def) = scene.update_for_command(
                    command_type, experiment_config, context)
                io_def.merge(scene_io_def)
                val_scenes.append(new_config)

            test_scenes = []
            for scene in self.test_scenes:
                if command_type == rv.PREDICT:
                    # Ensure there is a label store associated with
                    # predict and validation scenes.
                    if not scene.label_store:
                        scene = scene.to_builder() \
                                     .with_task(experiment_config.task) \
                                     .with_label_store() \
                                     .build()
                (new_config, scene_io_def) = scene.update_for_command(
                    command_type, experiment_config, context)
                io_def.merge(scene_io_def)
                test_scenes.append(new_config)
        else:
            test_scenes = self.test_scenes
            val_scenes = self.validation_scenes

        if command_type == rv.CHIP:
            augmentors = []
            for augmentor in self.augmentors:
                (new_config, aug_io_def) = augmentor.update_for_command(
                    command_type, experiment_config, context)
                io_def.merge(aug_io_def)
                augmentors.append(new_config)
        else:
            augmentors = self.augmentors

        conf = self.to_builder().with_train_scenes(train_scenes) \
                                .with_validation_scenes(val_scenes) \
                                .with_test_scenes(test_scenes) \
                                .with_augmentors(augmentors) \
                                .build()

        return (conf, io_def)

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

    def from_proto(self, msg):
        train_scenes = list(
            map(lambda x: SceneConfig.from_proto(x), msg.train_scenes))
        val_scenes = list(
            map(lambda x: SceneConfig.from_proto(x), msg.validation_scenes))
        test_scenes = list(
            map(lambda x: SceneConfig.from_proto(x), msg.test_scenes))
        augmentors = list(
            map(lambda x: AugmentorConfig.from_proto(x), msg.augmentors))
        return DatasetConfigBuilder() \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(val_scenes) \
            .with_test_scenes(test_scenes) \
            .with_augmentors(augmentors)

    def with_train_scenes(self, scenes):
        """Sets the scenes to be used for training."""
        b = deepcopy(self)
        b.config['train_scenes'] = scenes
        return b

    def with_train_scene(self, scene):
        """Sets the scene to be used for training."""
        return self.with_train_scenes([scene])

    def with_validation_scenes(self, scenes):
        """Sets the scenes to be used for validation."""
        b = deepcopy(self)
        b.config['validation_scenes'] = scenes
        return b

    def with_validation_scene(self, scene):
        """Sets the scene to be used for validation."""
        return self.with_validation_scenes([scene])

    def with_test_scenes(self, scenes):
        """Sets the scenes to be used for testing."""
        b = deepcopy(self)
        b.config['test_scenes'] = scenes
        return b

    def with_test_scene(self, scene):
        """Sets the scene to be used for testing."""
        return self.with_test_scenes([scene])

    def with_augmentors(self, augmentors):
        """Sets the data augmentors to be used."""
        b = deepcopy(self)
        b.config['augmentors'] = augmentors
        return b

    def with_augmentor(self, augmentor):
        """Sets the data augmentor to be used."""
        return self.with_augmentors([augmentor])
