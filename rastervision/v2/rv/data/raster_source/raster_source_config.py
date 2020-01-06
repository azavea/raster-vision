from typing import List

from rastervision.v2.core.config import Config

class RasterSourceConfig(Config):
    channel_order: List[int]
