from typing import List

from rastervision.v2.core.config import Config, register_config

@register_config('class_config')
class ClassConfig(Config):
    names: List[str]
    colors: List[str]

    def update(self, task=None):
        pass
