from typing import Optional, List

from rastervision.v2.core.config import Config, register_config
from rastervision.v2.rv.data.scene_config import SceneConfig
from rastervision.v2.rv.data.class_config import ClassConfig

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
