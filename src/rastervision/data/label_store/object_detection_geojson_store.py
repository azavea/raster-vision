import json

from rastervision.data.label import ObjectDetectionLabels
from rastervision.data.label_store import LabelStore
from rastervision.data.label_store.utils import boxes_to_geojson
from rastervision.data.label_source.utils import (
    load_label_store_json, geojson_to_object_detection_labels)
from rastervision.utils.files import str_to_file


class ObjectDetectionGeoJSONStore(LabelStore):
    def __init__(self, uri, crs_transformer, class_map):
        """Construct LabelStore backed by a GeoJSON file for object detection labels.

        Args:
            uri: uri of GeoJSON file containing labels
            crs_transformer: CRSTransformer to convert from map coords in label
                in GeoJSON file to pixel coords.
            class_map: ClassMap used to infer class_ids from class_name
                (or label) field
        """
        self.uri = uri
        self.crs_transformer = crs_transformer
        self.class_map = class_map

    def save(self, labels):
        """Save labels to URI."""
        boxes = labels.get_boxes()
        class_ids = labels.get_class_ids().tolist()
        scores = labels.get_scores().tolist()
        geojson_dict = boxes_to_geojson(
            boxes,
            class_ids,
            self.crs_transformer,
            self.class_map,
            scores=scores)
        geojson_str = json.dumps(geojson_dict)
        str_to_file(geojson_str, self.uri)

    def get_labels(self):
        self.labels = ObjectDetectionLabels.make_empty()

        json_dict = load_label_store_json(self.uri)
        return geojson_to_object_detection_labels(json_dict,
                                                  self.crs_transformer)

    def empty_labels(self):
        return ObjectDetectionLabels.make_empty()
