import json

from rv2.annotations.object_detection_annotations import (
    ObjectDetectionAnnotations)
from rv2.utils.files import file_to_str, str_to_file
from rv2.annotation_sources.object_detection_annotation_source import (
        ObjectDetectionAnnotationSource)


class GeoJSONFile(ObjectDetectionAnnotationSource):
    def __init__(self, uri, crs_transformer, writable=False):
        self.uri = uri
        self.crs_transformer = crs_transformer
        self.writable = writable

        try:
            geojson = json.loads(file_to_str(uri))
            self.annotations = ObjectDetectionAnnotations.from_geojson(
                geojson, crs_transformer)
        except:
            if writable:
                self.annotations = ObjectDetectionAnnotations.make_empty()
            else:
                raise ValueError('Could not open {}'.format(uri))

    def save(self, label_map):
        if self.writable:
            geojson = self.annotations.to_geojson(
                self.crs_transformer, label_map)
            geojson_str = json.dumps(geojson)
            str_to_file(geojson_str, self.uri)
        else:
            raise ValueError('Cannot save with writable=False')
