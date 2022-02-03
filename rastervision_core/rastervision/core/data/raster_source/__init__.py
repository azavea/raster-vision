# flake8: noqa

from rastervision.core.data.raster_source.raster_source import *
from rastervision.core.data.raster_source.raster_source_config import *
from rastervision.core.data.raster_source.rasterio_source import *
from rastervision.core.data.raster_source.rasterio_source_config import *
from rastervision.core.data.raster_source.rasterized_source import *
from rastervision.core.data.raster_source.rasterized_source_config import *
from rastervision.core.data.raster_source.multi_raster_source import *
from rastervision.core.data.raster_source.multi_raster_source_config import *


def register_plugin(registry):
    registry.set_plugin_version('rastervision.core', 1)
