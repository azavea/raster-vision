import json

import numpy as np

from rastervision.core import Box
from rastervision.data.label import ChipClassificationLabels
from rastervision.data.label_source.utils import load_label_store_json
from rastervision.data.label_store import LabelStore
from rastervision.data.label_store.utils import classification_labels_to_geojson
from rastervision.utils.files import str_to_file


def geojson_to_labels(geojson_dict, crs_transformer, extent=None):
    """Convert GeoJSON to ChipClassificationLabels from predictions.

    Args:
        geojson_dict: dict in GeoJSON format
        crs_transformer: used to convert map coords in geojson to pixel coords
            in labels object

    Returns:
       ChipClassificationLabels
    """
    features = geojson_dict['features']

    labels = ChipClassificationLabels()

    def polygon_to_label(polygon, crs_transformer):
        polygon = [crs_transformer.map_to_pixel(p) for p in polygon]
        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)
        cell = Box(ymin, xmin, ymax, xmax)

        properties = feature['properties']
        class_id = properties['class_id']
        scores = properties.get('scores')

        labels.set_cell(cell, class_id, scores)

    for feature in features:
        geom_type = feature['geometry']['type']
        coordinates = feature['geometry']['coordinates']
        if geom_type == 'Polygon':
            polygon_to_label(coordinates[0], crs_transformer)
        else:
            raise Exception(
                'Geometries of type {} are not supported in chip classification \
                labels.'.format(geom_type))
    return labels


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

        geojson_str = json.dumps(geojson_dict)

        str_to_file(geojson_str, self.uri)

    def get_labels(self):
        json_dict = load_label_store_json(self.uri)
        return geojson_to_labels(json_dict, self.crs_transformer)

    def empty_labels(self):
        return ChipClassificationLabels()
