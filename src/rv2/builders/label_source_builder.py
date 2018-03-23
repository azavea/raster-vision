from rv2.label_sources.object_detection_geojson_file import (
    ObjectDetectionGeoJSONFile)
from rv2.label_sources.classification_geojson_file import (
    ClassificationGeoJSONFile)


def build(config, crs_transformer, extent, writable=False):
    label_source_type = config.WhichOneof('label_source_type')
    if label_source_type == 'object_detection_geojson_file':
        return ObjectDetectionGeoJSONFile(
            config.object_detection_geojson_file.uri, crs_transformer,
            writable=writable)
    elif label_source_type == 'classification_geojson_file':
        return ClassificationGeoJSONFile(
            config.classification_geojson_file.uri, crs_transformer, extent,
            config.classification_geojson_file.options, writable=writable)
