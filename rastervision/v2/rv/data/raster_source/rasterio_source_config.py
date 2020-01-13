from typing import List

from rastervision.v2.rv.data.raster_source import RasterSourceConfig, RasterioSource
from rastervision.v2.core.config import register_config


@register_config('rasterio_source')
class RasterioSourceConfig(RasterSourceConfig):
    uris: List[str]
    x_shift: float = 0.0
    y_shift: float = 0.0

    def build(self, tmp_dir):
        # TODO
        raster_transformers = []
        return RasterioSource(
            self.uris,
            raster_transformers,
            tmp_dir,
            channel_order=self.channel_order,
            x_shift=self.x_shift,
            y_shift=self.y_shift)
