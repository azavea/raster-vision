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
