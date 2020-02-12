from typing import List

from rastervision2.core.data.raster_source import RasterSourceConfig, RasterioSource
from rastervision2.pipeline.config import register_config


@register_config('rasterio_source')
class RasterioSourceConfig(RasterSourceConfig):
    uris: List[str]
    x_shift: float = 0.0
    y_shift: float = 0.0

    def build(self, tmp_dir, use_transformers=True):
        raster_transformers = ([rt.build() for rt in self.transformers]
                               if use_transformers else [])

        return RasterioSource(
            self.uris,
            raster_transformers,
            tmp_dir,
            channel_order=self.channel_order,
            x_shift=self.x_shift,
            y_shift=self.y_shift)
