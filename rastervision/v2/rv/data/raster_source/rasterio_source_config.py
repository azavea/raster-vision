from typing import List

from rastervision.v2.rv.data.raster_source import RasterSourceConfig
from rastervision.v2.core.config import register_config

@register_config('rasterio_source')
class RasterioSourceConfig(RasterSourceConfig):
    uris: List[str]
    x_shift_meters: float = 0.0
    y_shift_meters: float = 0.0
