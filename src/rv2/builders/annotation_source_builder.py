from rv2.annotation_sources.geojson_file import GeoJSONFile


def build(config, crs_transformer, writable=False):
    # I wish crs_transformer wasn't required, but I don't see an easy way
    # around it right now.
    annotation_source_type = config.WhichOneof('annotation_source_type')
    if annotation_source_type == 'geojson_file':
        return GeoJSONFile(
            config.geojson_file.uri, crs_transformer, writable=writable)
