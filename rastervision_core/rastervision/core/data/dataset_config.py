from typing import Dict, List, Set

from rastervision.pipeline.config import (Config, register_config, ConfigError,
                                          Field)
from rastervision.pipeline.utils import split_into_groups
from rastervision.core.data.scene_config import SceneConfig
from rastervision.core.data.class_config import ClassConfig


def dataset_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version < 1:
        try:
            # removed in version 1
            del cfg_dict['img_channels']
        except KeyError:
            pass
    return cfg_dict


@register_config('dataset', upgrader=dataset_config_upgrader)
class DatasetConfig(Config):
    """Configure train, validation, and test splits for a dataset."""
    class_config: ClassConfig
    train_scenes: List[SceneConfig]
    validation_scenes: List[SceneConfig]
    test_scenes: List[SceneConfig] = []
    scene_groups: Dict[str, Set[str]] = Field(
        {},
        description='Groupings of scenes. Should be a dict of the form: '
        '{<group-name>: Set(scene_id_1, scene_id_2, ...)}. Three groups are '
        'added by default: "train_scenes", "validation_scenes", and '
        '"test_scenes"')

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

        # add default scene groups
        self.scene_groups['train_scenes'] = {s.id for s in self.train_scenes}
        self.scene_groups['test_scenes'] = {s.id for s in self.test_scenes}
        self.scene_groups['validation_scenes'] = {
            s.id
            for s in self.validation_scenes
        }

    def validate_config(self):
        ids = [s.id for s in self.train_scenes]
        if len(set(ids)) != len(ids):
            raise ConfigError('All training scene ids must be unique.')

        ids = [s.id for s in self.validation_scenes + self.test_scenes]
        if len(set(ids)) != len(ids):
            raise ConfigError(
                'All validation and test scene ids must be unique.')

        all_ids = {s.id for s in self.all_scenes}
        for group_name, group_ids in self.scene_groups.items():
            unknown_ids = group_ids.difference(all_ids)
            if len(unknown_ids) > 0:
                raise ConfigError(
                    f'IDs {unknown_ids} in scene group '
                    f'"{group_name}" do not match any scene in the dataset.')

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

    @property
    def all_scenes(self) -> List[SceneConfig]:
        return self.train_scenes + self.validation_scenes + self.test_scenes
