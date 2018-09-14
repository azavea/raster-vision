import json

from rastervision.data.label import ChipClassificationLabels
from rastervision.data.label_source.chip_classification_geojson_source \
    import read_labels
from rastervision.data.label_source.utils import load_label_store_json
from rastervision.data.label_store import LabelStore
from rastervision.data.label_store.utils import classification_labels_to_geojson
from rastervision.utils.files import str_to_file


class ChipClassificationGeoJSONStore(LabelStore):
    """A GeoJSON file with classification labels in it.

    Ideally the GeoJSON file contains a square for each cell in the grid. But
    in reality, it can be difficult to label imagery in such an exhaustive way.
    So, this can also handle GeoJSON files with non-overlapping polygons that
    do not necessarily cover the entire extent. It infers the grid of cells
    and associated class_ids using the extent and options if infer_cells is
    set to True.

    Args:
        options: ClassificationGeoJSONFile.Options
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
        # import pdb; pdb.set_trace()
        geojson_str = json.dumps(geojson_dict)

        str_to_file(geojson_str, self.uri)

    def get_labels(self):
        self.labels = ChipClassificationLabels()

        json_dict = load_label_store_json(self.uri)
        return read_labels(json_dict, self.crs_transformer)

    def empty_labels(self):
        return ChipClassificationLabels()

    def concatenate(self, labels1, labels2):
        labels1.extend(labels2)
        return labels1
