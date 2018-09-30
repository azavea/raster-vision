import json

from rastervision.data.label import ChipClassificationLabels
from rastervision.data.label_source.utils import load_label_store_json
from rastervision.data.label_store import LabelStore
from rastervision.data.label_source.utils import geojson_to_chip_classification_labels
from rastervision.data.label_store.utils import classification_labels_to_geojson

from rastervision.utils.files import str_to_file


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
        geojson_dict = classification_labels_to_geojson(
            labels, self.crs_transformer, self.class_map)
        geojson_str = json.dumps(geojson_dict)

        str_to_file(geojson_str, self.uri)

    def get_labels(self):
        json_dict = load_label_store_json(self.uri)
        return geojson_to_chip_classification_labels(json_dict,
                                                     self.crs_transformer)

    def empty_labels(self):
        return ChipClassificationLabels()
