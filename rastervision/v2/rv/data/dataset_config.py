from typing import Optional, List

from rastervision.v2.core import Config
from rastervision.v2.rv.data import SceneConfig

class DatasetConfig(Config):
    train_scenes: List[SceneConfig]
    validation_scenes: List[SceneConfig]
    test_scenes: Optional[List[SceneConfig]] = None
