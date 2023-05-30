from typing import TYPE_CHECKING, Optional
import logging

from rastervision.core.data.label import ObjectDetectionLabels
from rastervision.core.data.label_store import LabelStore
from rastervision.core.data.label_store.utils import boxes_to_geojson
from rastervision.core.data.vector_source import GeoJSONVectorSourceConfig
from rastervision.pipeline.file_system import json_to_file

if TYPE_CHECKING:
    from rastervision.core.box import Box
    from rastervision.core.data import ClassConfig, CRSTransformer

log = logging.getLogger(__name__)


class ObjectDetectionGeoJSONStore(LabelStore):
    """Storage for object detection predictions."""

    def __init__(self,
                 uri: str,
                 class_config: 'ClassConfig',
                 crs_transformer: 'CRSTransformer',
                 bbox: Optional['Box'] = None):
        """Constructor.

        Args:
            uri: uri of GeoJSON file containing labels
            class_config: ClassConfig used to infer class_ids from class_name
                (or label) field
            crs_transformer: CRSTransformer to convert from map coords in label
                in GeoJSON file to pixel coords.
            bbox (Optional[Box], optional): User-specified crop of the extent.
                If provided, only labels falling inside it are returned by
                :meth:`.ObjectDetectionGeoJSONStore.get_labels`.
        """
        self.uri = uri
        self.class_config = class_config
        self._crs_transformer = crs_transformer
        self._bbox = bbox

    def save(self, labels: ObjectDetectionLabels) -> None:
        """Save labels to URI."""
        log.info(f'Saving {len(labels)} boxes as GeoJSON.')
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
        vector_source = GeoJSONVectorSourceConfig(uris=self.uri).build(
            class_config=self.class_config,
            crs_transformer=self.crs_transformer)
        labels = ObjectDetectionLabels.from_geojson(
            vector_source.get_geojson())
        if self.bbox is not None:
            labels = ObjectDetectionLabels.get_overlapping(labels, self.bbox)
        return labels

    @property
    def bbox(self) -> 'Box':
        return self._bbox

    @property
    def crs_transformer(self) -> 'CRSTransformer':
        return self._crs_transformer

    def set_bbox(self, bbox: 'Box') -> None:
        self._bbox = bbox
