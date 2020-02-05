from typing import Optional, List

from rastervision2.pipeline.config import Config, register_config
from rastervision2.core.data.scene_config import SceneConfig
from rastervision2.core.data.class_config import ClassConfig
from rastervision2.core.utils.misc import split_into_groups


@register_config('dataset')
class DatasetConfig(Config):
    class_config: ClassConfig
    train_scenes: List[SceneConfig]
    validation_scenes: List[SceneConfig]
    test_scenes: Optional[List[SceneConfig]] = None

    def update(self, task=None):
        super().update()

        self.class_config.update(task=task)
        for s in self.train_scenes:
            s.update(task=task)
        for s in self.validation_scenes:
            s.update(task=task)
        if self.test_scenes is not None:
            for s in self.test_scenes:
                s.update(task=task)

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
