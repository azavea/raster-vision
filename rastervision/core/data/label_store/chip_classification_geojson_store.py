from rastervision.pipeline.file_system import json_to_file
from rastervision.core.data.label import ChipClassificationLabels
from rastervision.core.data.label_store import LabelStore
from rastervision.core.data.label_store.utils import boxes_to_geojson
from rastervision.core.data.label_source import (
    ChipClassificationLabelSourceConfig)
from rastervision.core.data.vector_source import (GeoJSONVectorSourceConfig)


class ChipClassificationGeoJSONStore(LabelStore):
    """Storage for chip classification predictions."""

    def __init__(self, uri, class_config, crs_transformer):
        """Constructor.

        Args:
            uri: uri of GeoJSON file containing labels
            class_config: ClassConfig
            crs_transformer: CRSTransformer to convert from map coords in label
                in GeoJSON file to pixel coords.
        """
        self.uri = uri
        self.class_config = class_config
        self.crs_transformer = crs_transformer

    def save(self, labels):
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

    def get_labels(self):
        vs = GeoJSONVectorSourceConfig(uri=self.uri, default_class_id=None)
        ls = ChipClassificationLabelSourceConfig(vector_source=vs).build(
            self.class_config, self.crs_transformer)
        return ls.get_labels()

    def empty_labels(self):
        return ChipClassificationLabels()
