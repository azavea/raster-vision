from typing import List

from rastervision2.pipeline.config import Config, register_config


@register_config('raster_source')
class RasterSourceConfig(Config):
    channel_order: List[int]

    def build(self, tmp_dir):
        raise NotImplementedError()

    def update(self, pipeline=None, scene=None):
        pass
