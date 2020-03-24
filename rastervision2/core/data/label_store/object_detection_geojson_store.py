from rastervision2.core.data.label import ObjectDetectionLabels
from rastervision2.core.data.label_store import LabelStore
from rastervision2.core.data.label_store.utils import boxes_to_geojson
from rastervision2.core.data.vector_source import GeoJSONVectorSourceConfig
from rastervision2.pipeline.file_system import json_to_file


class ObjectDetectionGeoJSONStore(LabelStore):
    def __init__(self, uri, class_config, crs_transformer):
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
        self.class_config = class_config

    def save(self, labels):
        """Save labels to URI."""
        boxes = labels.get_boxes()
        class_ids = labels.get_class_ids().tolist()
        scores = labels.get_scores().tolist()
        geojson = boxes_to_geojson(
            boxes,
            class_ids,
            self.crs_transformer,
            self.class_config,
            scores=scores)
        json_to_file(geojson, self.uri)

    def get_labels(self):
        vector_source = GeoJSONVectorSourceConfig(uri=self.uri).build(
            self.class_config, self.crs_transformer)
        return ObjectDetectionLabels.from_geojson(vector_source.get_geojson())

    def empty_labels(self):
        return ObjectDetectionLabels.make_empty()
