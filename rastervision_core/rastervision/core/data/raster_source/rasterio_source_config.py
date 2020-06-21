from typing import List

from rastervision.core.data.raster_source import RasterSourceConfig, RasterioSource
from rastervision.pipeline.config import register_config, Field


@register_config('rasterio_source')
class RasterioSourceConfig(RasterSourceConfig):
    uris: List[str] = Field(
        ...,
        description=
        ('List of image URIs that comprise imagery for a scene. The format of each file '
         'can be any that can be read by Rasterio/GDAL. If > 1 URI is provided, a VRT '
         'will be created to mosaic together the individual images.'))
    x_shift: float = Field(
        0.0,
        descriptions=
        ('A number of meters to shift along the x-axis. A positive shift moves the '
         '"camera" to the right.'))
    y_shift: float = Field(
        0.0,
        descriptions=
        ('A number of meters to shift along the y-axis. A positive shift moves the '
         '"camera" down.'))

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
