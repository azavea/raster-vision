import json

from rastervision.labels.object_detection_labels import (
    ObjectDetectionLabels)
from rastervision.label_stores.utils import add_classes_to_geojson
from rastervision.utils.files import file_to_str, str_to_file
from rastervision.label_stores.object_detection_label_store import (
        ObjectDetectionLabelStore)


class ObjectDetectionGeoJSONFile(ObjectDetectionLabelStore):
    # TODO allow null crs_transformer for when we assume that the labels
    # are already in the crs and don't need to be converted.
    def __init__(self, uri, crs_transformer, class_map, writable=False):
        self.uri = uri
        self.crs_transformer = crs_transformer
        self.class_map = class_map
        self.writable = writable

        try:
            geojson = json.loads(file_to_str(uri))
            geojson = add_classes_to_geojson(geojson, class_map)
            self.labels = ObjectDetectionLabels.from_geojson(
                geojson, crs_transformer)
        except:
            if self.writable or not self.uri:
                self.labels = ObjectDetectionLabels.make_empty()
            else:
                raise ValueError('Could not open {}'.format(uri))

    def save(self):
        if self.writable:
            geojson = self.labels.to_geojson(
                self.crs_transformer, self.class_map)
            geojson_str = json.dumps(geojson)
            str_to_file(geojson_str, self.uri)
        else:
            raise ValueError('Cannot save with writable=False')
