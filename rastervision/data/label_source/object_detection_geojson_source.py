from rastervision.data.label import ObjectDetectionLabels
from rastervision.data.label_source import LabelSource
from rastervision.data.label_source.utils import (
    add_classes_to_geojson, load_label_store_json,
    geojson_to_object_detection_labels)


class ObjectDetectionGeoJSONSource(LabelSource):
    def __init__(self, uri, crs_transformer, class_map, extent=None):
        """Construct ObjectDetectionLabelStore backed by a GeoJSON file.

        Args:
            uri: uri of GeoJSON file containing labels
            crs_transformer: CRSTransformer to convert from map coords in label
                in GeoJSON file to pixel coords.
            class_map: ClassMap used to infer class_ids from class_name
                (or label) field
            extent: Box used to filter the labels by extent
        """
        self.labels = ObjectDetectionLabels.make_empty()

        json_dict = load_label_store_json(uri)
        if json_dict:
            geojson = add_classes_to_geojson(json_dict, class_map)
            self.labels = geojson_to_object_detection_labels(
                geojson, crs_transformer, extent=extent)

    def get_labels(self, window=None):
        if window is None:
            return self.labels

        return ObjectDetectionLabels.get_overlapping(self.labels, window)
