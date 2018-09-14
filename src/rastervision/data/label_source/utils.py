import copy
import json

import numpy as np

from rastervision.core.box import Box
from rastervision.data import ObjectDetectionLabels
from rastervision.utils.files import file_to_str


def boxes_to_geojson(boxes, class_ids, crs_transformer, class_map,
                     scores=None):
    """Convert boxes and associated data into a GeoJSON dict.

    Args:
        boxes: list of Box in pixel row/col format.
        class_ids: list of int (one for each box)
        crs_transformer: CRSTransformer used to convert pixel coords to map
            coords in the GeoJSON
        class_map: ClassMap used to infer class_name from class_id
        scores: optional list of floats (one for each box)


    Returns:
        dict in GeoJSON format
    """
    features = []
    for box_ind, box in enumerate(boxes):
        polygon = box.geojson_coordinates()
        polygon = [list(crs_transformer.pixel_to_map(p)) for p in polygon]

        class_id = int(class_ids[box_ind])
        class_name = class_map.get_by_id(class_id).name
        score = 0.0
        if scores is not None:
            score = scores[box_ind]

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            },
            'properties': {
                'class_id': class_id,
                'class_name': class_name,
                'score': score
            }
        }
        features.append(feature)

    return {'type': 'FeatureCollection', 'features': features}


def add_classes_to_geojson(geojson, class_map):
    """Add missing class_names and class_ids from label GeoJSON."""
    geojson = copy.deepcopy(geojson)
    features = geojson['features']

    for feature in features:
        properties = feature.get('properties', {})
        if 'class_id' not in properties:
            if 'class_name' in properties:
                properties['class_id'] = \
                    class_map.get_by_name(properties['class_name']).id
            elif 'label' in properties:
                # label is considered a synonym of class_name for now in order
                # to interface with Raster Foundry.
                properties['class_id'] = \
                    class_map.get_by_name(properties['label']).id
                properties['class_name'] = properties['label']
            else:
                # if no class_id, class_name, or label, then just assume
                # everything corresponds to class_id = 1.
                class_id = 1
                class_name = class_map.get_by_id(class_id).name
                properties['class_id'] = class_id
                properties['class_name'] = class_name

        feature['properties'] = properties

    return geojson


def load_label_store_json(uri):
    """Load JSON for LabelStore.

    Returns JSON for uri
    """
    return json.loads(file_to_str(uri))


def geojson_to_object_detection_labels(geojson_dict,
                                       crs_transformer,
                                       extent=None):
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
                'Geometries of type {} are not supported in object detection \
                labels.'.format(geom_type))

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
