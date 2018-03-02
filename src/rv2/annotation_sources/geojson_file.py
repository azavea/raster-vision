import json

from rv2.annotations.object_detection_annotations import (
    ObjectDetectionAnnotations)
from rv2.utils.files import file_to_str, str_to_file
from rv2.core.annotation_source import AnnotationSource


class GeoJSONFile(AnnotationSource):
    def __init__(self, uri, crs_transformer, write_mode=False):
        self.uri = uri
        self.crs_transformer = crs_transformer
        self.write_mode = write_mode

        if write_mode:
            self.annotations = ObjectDetectionAnnotations.make_empty()
        else:
            geojson = json.loads(file_to_str(uri))
            self.annotations = ObjectDetectionAnnotations.from_geojson(
                geojson, crs_transformer)

    def get_annotations(self, window, ioa_thresh=1.0):
        return self.annotations.get_subset(
            window, ioa_thresh=ioa_thresh)

    def get_all_annotations(self):
        return self.annotations

    def extend(self, window, annotations):
        self.annotations = self.annotations.concatenate(
            window, annotations)

    def post_process(self, options):
        self.annotations = self.annotations.prune_duplicates(
            options.object_detection_options.score_thresh,
            options.object_detection_options.merge_thresh)

    def save(self, label_map):
        if self.write_mode:
            geojson = self.annotations.to_geojson(
                self.crs_transformer, label_map)
            geojson_str = json.dumps(geojson)
            str_to_file(geojson_str, self.uri)
        else:
            raise ValueError('Cannot save with write_mode=False')
