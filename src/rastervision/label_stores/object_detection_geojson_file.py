import json

import numpy as np

from rastervision.core.box import Box
from rastervision.labels.object_detection_labels import (ObjectDetectionLabels)
from rastervision.label_stores.utils import (add_classes_to_geojson,
                                             load_label_store_json)
from rastervision.label_stores.object_detection_label_store import (
    ObjectDetectionLabelStore)
from rastervision.label_stores.utils import boxes_to_geojson
from rastervision.utils.files import str_to_file, NotWritableError


def geojson_to_labels(geojson_dict, crs_transformer, extent=None):
    """Convert GeoJSON to ObjectDetectionLabels object.

    If extent is provided, filter out the boxes that lie "more than a little
    bit" outside the extent.

    Args:
        geojson_dict: dict in GeoJSON format
        crs_transformer: used to convert map coords in geojson to pixel coords
            in labels object
        extent: Box in pixel coords

    Returns:
        ObjectDetectionLabels
    """
    features = geojson_dict['features']
    boxes = []
    class_ids = []
    scores = []

    def polygon_to_label(polygon, crs_transformer):
        polygon = [crs_transformer.map_to_pixel(p) for p in polygon]
        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)
        boxes.append(Box(ymin, xmin, ymax, xmax))

        properties = feature['properties']
        class_ids.append(properties['class_id'])
        scores.append(properties.get('score', 1.0))

    for feature in features:
        geom_type = feature['geometry']['type']
        coordinates = feature['geometry']['coordinates']
        if geom_type == 'MultiPolygon':
            for polygon in coordinates:
                polygon_to_label(polygon[0], crs_transformer)
        elif geom_type == 'Polygon':
            polygon_to_label(coordinates[0], crs_transformer)
        else:
            raise Exception(
                "Geometries of type {} are not supported in object detection \
                labels.".format(geom_type))

    if len(boxes):
        boxes = np.array([box.npbox_format() for box in boxes], dtype=float)
        class_ids = np.array(class_ids)
        scores = np.array(scores)
        labels = ObjectDetectionLabels(boxes, class_ids, scores=scores)
    else:
        labels = ObjectDetectionLabels.make_empty()

    if extent is not None:
        labels = ObjectDetectionLabels.get_overlapping(
            labels, extent, ioa_thresh=0.8, clip=True)
    return labels


class ObjectDetectionGeoJSONFile(ObjectDetectionLabelStore):
    def __init__(self,
                 uri,
                 crs_transformer,
                 class_map,
                 extent=None,
                 readable=True,
                 writable=False):
        """Construct ObjectDetectionLabelStore backed by a GeoJSON file.

        Args:
            uri: uri of GeoJSON file containing labels
            crs_transformer: CRSTransformer to convert from map coords in label
                in GeoJSON file to pixel coords.
            class_map: ClassMap used to infer class_ids from class_name
                (or label) field
            extent: Box used to filter the labels by extent
            readable: if True, expect the file to exist
            writable: if True, allow writing to disk
        """
        self.uri = uri
        self.crs_transformer = crs_transformer
        self.class_map = class_map
        self.readable = readable
        self.writable = writable

        self.labels = ObjectDetectionLabels.make_empty()

        json_dict = load_label_store_json(uri, readable)
        if json_dict:
            geojson = add_classes_to_geojson(json_dict, class_map)
            self.labels = geojson_to_labels(
                geojson, crs_transformer, extent=extent)

    def save(self):
        """Save labels to URI if writable."""
        if self.writable:
            boxes = self.labels.get_boxes()
            class_ids = self.labels.get_class_ids().tolist()
            scores = self.labels.get_scores().tolist()
            geojson_dict = boxes_to_geojson(
                boxes,
                class_ids,
                self.crs_transformer,
                self.class_map,
                scores=scores)
            geojson_str = json.dumps(geojson_dict)
            str_to_file(geojson_str, self.uri)
        else:
            raise NotWritableError('Cannot write with writable=False')
