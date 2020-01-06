from typing import Optional, List

from rastervision.v2.core import Config
from rastervision.v2.rv.data import (SceneConfig, ClassConfig)

class DatasetConfig(Config):
    class_config: ClassConfig
    train_scenes: List[SceneConfig]
    validation_scenes: List[SceneConfig]
    test_scenes: Optional[List[SceneConfig]] = None
