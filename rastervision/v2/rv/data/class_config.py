from typing import List

from rastervision.v2.core import Config

class ClassConfig(Config):
    names: List[str]
    colors: List[str]
