from typing import TYPE_CHECKING, Optional

from rastervision.pipeline.file_system import json_to_file
from rastervision.core.data.label import ChipClassificationLabels
from rastervision.core.data.label_store import LabelStore
from rastervision.core.data.label_store.utils import boxes_to_geojson
from rastervision.core.data.label_source import (
    ChipClassificationLabelSourceConfig)
from rastervision.core.data.vector_source import (GeoJSONVectorSourceConfig)

if TYPE_CHECKING:
    from rastervision.core.box import Box
    from rastervision.core.data import ClassConfig, CRSTransformer


class ChipClassificationGeoJSONStore(LabelStore):
    """Storage for chip classification predictions."""

    def __init__(self,
                 uri: str,
                 class_config: 'ClassConfig',
                 crs_transformer: 'CRSTransformer',
                 bbox: Optional['Box'] = None):
        """Constructor.

        Args:
            uri: uri of GeoJSON file containing labels
            class_config: ClassConfig
            crs_transformer: CRSTransformer to convert from map coords in label
                in GeoJSON file to pixel coords.
            bbox (Optional[Box], optional): User-specified crop of the extent.
                If provided, only labels falling inside it are returned by
                :meth:`.ChipClassificationGeoJSONStore.get_labels`.
        """
        self.uri = uri
        self.class_config = class_config
        self._crs_transformer = crs_transformer
        self._bbox = bbox

    def save(self, labels: ChipClassificationLabels) -> None:
        """Save labels to URI if writable.

        Note that if the grid is inferred from polygons, only the grid will be
        written, not the original polygons.
        """
        boxes = labels.get_cells()
        class_ids = labels.get_class_ids()
        scores = list(labels.get_scores())
        geojson = boxes_to_geojson(
            boxes,
            class_ids,
            self.crs_transformer,
            self.class_config,
            scores=scores)
        json_to_file(geojson, self.uri)

    def get_labels(self) -> ChipClassificationLabels:
        vs = GeoJSONVectorSourceConfig(uris=self.uri)
        ls = ChipClassificationLabelSourceConfig(vector_source=vs).build(
            class_config=self.class_config,
            crs_transformer=self.crs_transformer,
            bbox=self.bbox)
        return ls.get_labels()

    @property
    def bbox(self) -> 'Box':
        return self._bbox

    @property
    def crs_transformer(self) -> 'CRSTransformer':
        return self._crs_transformer

    def set_bbox(self, bbox: 'Box') -> None:
        self._bbox = bbox
