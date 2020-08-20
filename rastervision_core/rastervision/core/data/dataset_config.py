from typing import List, Optional
from pydantic import PositiveInt

from rastervision.pipeline.config import (Config, register_config, ConfigError,
                                          Field)
from rastervision.pipeline.utils import split_into_groups
from rastervision.core.data.scene_config import SceneConfig
from rastervision.core.data.class_config import ClassConfig
from rastervision.core.data.utils import all_equal


@register_config('dataset')
class DatasetConfig(Config):
    """Config for a Dataset comprising the scenes for train, valid, and test splits."""
    class_config: ClassConfig
    train_scenes: List[SceneConfig]
    validation_scenes: List[SceneConfig]
    test_scenes: List[SceneConfig] = []

    img_channels: Optional[PositiveInt] = Field(
        None, description='The number of channels of the images.')

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

        channel_order = self.train_scenes[0].raster_source.channel_order
        if channel_order is not None:
            self.img_channels = len(channel_order)

    def validate_config(self):
        ids = [s.id for s in self.train_scenes]
        if len(set(ids)) != len(ids):
            raise ConfigError('All training scene ids must be unique.')

        ids = [s.id for s in self.validation_scenes + self.test_scenes]
        if len(set(ids)) != len(ids):
            raise ConfigError(
                'All validation and test scene ids must be unique.')

        self.ensure_same_channel_order()

    def ensure_same_channel_order(self):
        all_scenes = self.train_scenes + self.validation_scenes + self.test_scenes
        channel_orders = [s.raster_source.channel_order for s in all_scenes]
        if not all_equal(channel_orders):
            raise ConfigError('channel_order must be same for all scenes.')

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
