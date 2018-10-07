import copy
import json
from typing import Tuple

import numpy as np
from PIL import ImageColor

from rastervision.core.box import Box
from rastervision.data import (ChipClassificationLabels, ObjectDetectionLabels)
from rastervision.utils.files import file_to_str


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


def geojson_to_chip_classification_labels(geojson_dict,
                                          crs_transformer,
                                          extent=None):
    """Convert GeoJSON to ChipClassificationLabels.

    If extent is given, only labels that intersect with the extent are returned.

    Args:
        geojson_dict: dict in GeoJSON format
        crs_transformer: used to convert map coords in geojson to pixel coords
            in labels object
        extent: Box in pixel coords

    Returns:
       ChipClassificationLabels
    """
    features = geojson_dict['features']

    labels = ChipClassificationLabels()

    extent_shape = None
    if extent:
        extent_shape = extent.to_shapely()

    def polygon_to_label(polygon, crs_transformer):
        polygon = [crs_transformer.map_to_pixel(p) for p in polygon]
        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)
        cell = Box(ymin, xmin, ymax, xmax)

        if extent_shape and not cell.to_shapely().intersects(extent_shape):
            return

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


def color_to_triple(color: str) -> Tuple[int, int, int]:
    """Given a PIL ImageColor string, return a triple of integers
    representing the red, green, and blue values.

    Args:
         color: A PIL ImageColor string

    Returns:
         An triple of integers

    """
    if color is None:
        r = np.random.randint(0, 0x100)
        g = np.random.randint(0, 0x100)
        b = np.random.randint(0, 0x100)
        return (r, g, b)
    else:
        return ImageColor.getrgb(color)


def color_to_integer(color: str) -> int:
    """Given a PIL ImageColor string, return a packed integer.

    Args:
         color: A PIL ImageColor string

    Returns:
         An integer containing the packed RGB values.

    """
    triple = color_to_triple(color)
    r = triple[0] * (1 << 16)
    g = triple[1] * (1 << 8)
    b = triple[2] * (1 << 0)
    integer = r + g + b
    return integer


def rgb_to_int_array(rgb_array):
    r = np.array(rgb_array[:, :, 0], dtype=np.uint32) * (1 << 16)
    g = np.array(rgb_array[:, :, 1], dtype=np.uint32) * (1 << 8)
    b = np.array(rgb_array[:, :, 2], dtype=np.uint32) * (1 << 0)
    return r + g + b
