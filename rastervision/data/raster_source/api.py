# flake8: noqa

from rastervision.data.vector_source.api import GEOJSON_SOURCE

# Registry Keys
RASTER_SOURCE = 'RASTER_SOURCE'

GEOTIFF_SOURCE = 'GEOTIFF_SOURCE'
IMAGE_SOURCE = 'IMAGE_SOURCE'
RASTERIZED_SOURCE = 'RASTERIZED_SOURCE'

raster_source_deprecated_map = {GEOJSON_SOURCE: RASTERIZED_SOURCE}

from .raster_source_config import RasterSourceConfig
