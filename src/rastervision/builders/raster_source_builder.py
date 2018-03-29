from rastervision.raster_sources.geotiff_files import GeoTiffFiles
from rastervision.builders import raster_transformer_builder


def build(config):
    raster_transformer = raster_transformer_builder.build(
        config.raster_transformer)

    raster_source_type = config.WhichOneof('raster_source_type')
    if raster_source_type == 'geotiff_files':
        return GeoTiffFiles(raster_transformer, config.geotiff_files.uris)
