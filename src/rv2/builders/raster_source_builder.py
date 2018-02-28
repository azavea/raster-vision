from rv2.raster_sources.geotiff_files import GeoTiffFiles


def build(config):
    raster_source_type = config.WhichOneof('raster_source_type')
    if raster_source_type == 'geotiff_files':
        return GeoTiffFiles(config.geotiff_files.uris)
