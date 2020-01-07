from typing import List

from rastervision.v2.core.config import Config, register_config

@register_config('raster_source')
class RasterSourceConfig(Config):
    channel_order: List[int]

    def build(self, tmp_dir):
        raise NotImplementedError()

    def update(self, task=None, scene=None):
        pass
