from typing import TYPE_CHECKING

from rastervision.core.data.label import ObjectDetectionLabels
from rastervision.core.data.label_store import LabelStore
from rastervision.core.data.label_store.utils import boxes_to_geojson
from rastervision.core.data.vector_source import GeoJSONVectorSourceConfig
from rastervision.pipeline.file_system import json_to_file

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig, CRSTransformer


class ObjectDetectionGeoJSONStore(LabelStore):
    """Storage for object detection predictions."""

    def __init__(self, uri: str, class_config: 'ClassConfig',
                 crs_transformer: 'CRSTransformer'):
        """Constructor.

        Args:
            uri: uri of GeoJSON file containing labels
            class_config: ClassConfig used to infer class_ids from class_name
                (or label) field
            crs_transformer: CRSTransformer to convert from map coords in label
                in GeoJSON file to pixel coords.
        """
        self.uri = uri
        self.crs_transformer = crs_transformer
        self.class_config = class_config

    def save(self, labels: ObjectDetectionLabels) -> None:
        """Save labels to URI."""
        boxes = labels.get_boxes()
        class_ids = [int(id) for id in labels.get_class_ids()]
        scores = labels.get_scores()
        geojson = boxes_to_geojson(
            boxes,
            class_ids,
            self.crs_transformer,
            self.class_config,
            scores=scores)
        json_to_file(geojson, self.uri)

    def get_labels(self) -> ObjectDetectionLabels:
        vector_source = GeoJSONVectorSourceConfig(uri=self.uri).build(
            self.class_config, self.crs_transformer)
        return ObjectDetectionLabels.from_geojson(vector_source.get_geojson())

    def empty_labels(self) -> ObjectDetectionLabels:
        return ObjectDetectionLabels.make_empty()
