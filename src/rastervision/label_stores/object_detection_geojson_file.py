import json

from rastervision.labels.object_detection_labels import (
    ObjectDetectionLabels)
from rastervision.label_stores.utils import (
    add_classes_to_geojson, load_label_store_json)
from rastervision.utils.files import str_to_file
from rastervision.label_stores.object_detection_label_store import (
    ObjectDetectionLabelStore)


class ObjectDetectionGeoJSONFile(ObjectDetectionLabelStore):
    def __init__(self, uri, crs_transformer, extent, class_map,
                 readable=True, writable=False):
        self.uri = uri
        self.crs_transformer = crs_transformer
        self.class_map = class_map
        self.readable = readable
        self.writable = writable

        self.labels = ObjectDetectionLabels.make_empty()

        json_dict = load_label_store_json(uri, readable)
        if json_dict:
            geojson = add_classes_to_geojson(json_dict, class_map)
            self.labels = ObjectDetectionLabels.from_geojson(
                geojson, crs_transformer, extent)

    def save(self):
        if self.writable:
            geojson = self.labels.to_geojson(
                self.crs_transformer, self.class_map)
            geojson_str = json.dumps(geojson)
            str_to_file(geojson_str, self.uri)
        else:
            raise ValueError('Cannot save with writable=False')
