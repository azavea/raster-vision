from rv2.label_sources.geojson_file import GeoJSONFile


def build(config, crs_transformer, writable=False):
    # I wish crs_transformer wasn't required, but I don't see an easy way
    # around it right now.
    label_source_type = config.WhichOneof('label_source_type')
    if label_source_type == 'geojson_file':
        return GeoJSONFile(
            config.geojson_file.uri, crs_transformer, writable=writable)
