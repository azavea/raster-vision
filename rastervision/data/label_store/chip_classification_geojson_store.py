from rastervision.data.label import ChipClassificationLabels
from rastervision.data.label_store import LabelStore
from rastervision.data.label_store.utils import boxes_to_geojson
from rastervision.data.label_source import ChipClassificationLabelSource
from rastervision.data.vector_source import GeoJSONVectorSource
from rastervision.utils.files import json_to_file


class ChipClassificationGeoJSONStore(LabelStore):
    """A GeoJSON file with classification labels in it.
    """

    def __init__(self, uri, crs_transformer, class_map):
        """Construct ClassificationLabelStore backed by a GeoJSON file.

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
            self.class_map,
            scores=scores)
        json_to_file(geojson, self.uri)

    def get_labels(self):
        vector_source = GeoJSONVectorSource(self.uri, self.crs_transformer)
        source = ChipClassificationLabelSource(
            vector_source, self.crs_transformer, self.class_map)
        return source.get_labels()

    def empty_labels(self):
        return ChipClassificationLabels()
