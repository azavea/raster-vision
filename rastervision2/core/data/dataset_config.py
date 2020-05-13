from typing import List

from rastervision2.pipeline.config import Config, register_config, ConfigError
from rastervision2.pipeline.utils import split_into_groups
from rastervision2.core.data.scene_config import SceneConfig
from rastervision2.core.data.class_config import ClassConfig


@register_config('dataset')
class DatasetConfig(Config):
    class_config: ClassConfig
    train_scenes: List[SceneConfig]
    validation_scenes: List[SceneConfig]
    test_scenes: List[SceneConfig] = []

    def update(self, pipeline=None):
        super().update()

        self.class_config.update(pipeline=pipeline)
        for s in self.train_scenes:
            s.update(pipeline=pipeline)
        for s in self.validation_scenes:
            s.update(pipeline=pipeline)
        if self.test_scenes is not None:
            for s in self.test_scenes:
                s.update(pipeline=pipeline)

    def validate_config(self):
        ids = [s.id for s in self.train_scenes]
        if len(set(ids)) != len(ids):
            raise ConfigError('All training scene ids must be unique.')

        ids = [s.id for s in self.validation_scenes + self.test_scenes]
        if len(set(ids)) != len(ids):
            raise ConfigError(
                'All validation and test scene ids must be unique.')

    def get_split_config(self, split_ind, num_splits):
        new_cfg = self.copy()

        groups = split_into_groups(self.train_scenes, num_splits)
        new_cfg.train_scenes = groups[
            split_ind] if split_ind < len(groups) else []

        groups = split_into_groups(self.validation_scenes, num_splits)
        new_cfg.validation_scenes = groups[
            split_ind] if split_ind < len(groups) else []

        if self.test_scenes:
            groups = split_into_groups(self.test_scenes, num_splits)
            new_cfg.test_scenes = groups[
                split_ind] if split_ind < len(groups) else []

        return new_cfg

    def get_all_scenes(self):
        return self.train_scenes + self.validation_scenes + self.test_scenes
