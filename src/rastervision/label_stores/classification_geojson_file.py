import json

import numpy as np
from shapely.strtree import STRtree
from shapely import geometry

from rastervision.labels.classification_labels import (
    ClassificationLabels)
from rastervision.labels.object_detection_labels import ObjectDetectionLabels
from rastervision.labels.utils import boxes_to_geojson
from rastervision.label_stores.utils import add_classes_to_geojson
from rastervision.utils.files import file_to_str, str_to_file
from rastervision.label_stores.classification_label_store import (
        ClassificationLabelStore)


def get_str_tree(geojson, crs_transformer):
    features = geojson['features']
    json_polygons = []
    class_ids = []

    for feature in features:
        # Convert polygon to pixel coords.
        polygon = feature['geometry']['coordinates'][0]
        polygon = [crs_transformer.web_to_pixel(p) for p in polygon]
        json_polygons.append(polygon)

        properties = feature.get('properties', {})
        class_ids.append(properties.get('class_id', 1))

    # Convert polygons to shapely
    polygons = []
    for json_polygon, class_id in zip(json_polygons, class_ids):
        polygon = geometry.Polygon([(p[0], p[1]) for p in json_polygon])
        # Trick to handle self-intersecting polygons which otherwise cause an
        # error.
        polygon = polygon.buffer(0)
        polygon.class_id = class_id
        polygons.append(polygon)

    return STRtree(polygons)


def infer_labels(geojson, crs_transformer, extent, options):
    """Infer the class_id for each grid cell from polygons."""
    str_tree = get_str_tree(geojson, crs_transformer)
    labels = ClassificationLabels()

    # For each cell, find intersecting polygons.
    cells = extent.get_windows(options.cell_size, options.cell_size)
    for cell in cells:
        cell_geom = geometry.Polygon(
            [(p[0], p[1]) for p in cell.geojson_coordinates()])
        intersecting_polygons = str_tree.query(cell_geom)

        intersection_over_cells = []
        intersection_over_polygons = []
        class_ids = []

        # Find polygons whose intersection with the cell is big enough.
        for polygon in intersecting_polygons:
            intersection = polygon.intersection(cell_geom)
            intersection_over_cell = intersection.area / cell_geom.area
            intersection_over_polygon = intersection.area / polygon.area

            if options.use_intersection_over_cell:
                enough_intersection = intersection_over_cell >= options.ioa_thresh
            else:
                enough_intersection = intersection_over_polygon >= options.ioa_thresh

            if enough_intersection:
                intersection_over_cells.append(intersection_over_cell)
                intersection_over_polygons.append(intersection_over_polygon)
                class_ids.append(polygon.class_id)

        # Infer class id for cell.
        if len(class_ids) == 0:
            class_id = (None if options.background_class_id == 0
                        else options.background_class_id)
        elif options.pick_min_class_id:
            class_id = min(class_ids)
        else:
            # Pick class_id of the polygon with the biggest intersection over
            # cell.
            class_id = class_ids[np.argmax(intersection_over_cells)]

        labels.set_cell(cell, class_id)
    return labels


def convert_labels(od_labels, extent, options):
    labels = ClassificationLabels()
    boxes = od_labels.get_boxes()
    class_ids = od_labels.get_class_ids()

    for box, class_id in zip(boxes, class_ids):
        labels.set_cell(box, class_id)

    return labels


def load_geojson(geojson, crs_transformer, extent, options):
    """Construct ClassificationLabels from GeoJSON.

    Args:
        options: ClassificationGeoJSONFile.Options
    """
    if options.infer_cells:
        labels = infer_labels(geojson, crs_transformer, extent, options)
    else:
        # Use the ObjectDetectionLabels to parse bounding boxes out of the
        # GeoJSON.
        od_labels = ObjectDetectionLabels.from_geojson(
            geojson, crs_transformer)
        labels = convert_labels(od_labels, extent, options)

    return labels


def to_geojson(labels, crs_transformer, class_map):
    boxes = labels.get_cells()
    class_ids = labels.get_class_ids()

    return boxes_to_geojson(
        boxes, class_ids, crs_transformer, class_map)


class ClassificationGeoJSONFile(ClassificationLabelStore):
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
    def __init__(self, uri, crs_transformer, extent, options, class_map,
                 writable=False):
        self.uri = uri
        self.crs_transformer = crs_transformer
        self.class_map = class_map
        self.writable = writable

        self.set_grid(extent, options.cell_size)

        try:
            geojson = json.loads(file_to_str(uri))
            geojson = add_classes_to_geojson(geojson, class_map)
            self.labels = load_geojson(
                geojson, crs_transformer, extent, options)
        except:  # TODO do a better job of only catching "not found" errors
            if writable:
                self.labels = ClassificationLabels()
            else:
                raise ValueError('Could not open {}'.format(uri))

    def save(self):
        if self.writable:
            geojson = to_geojson(
                self.labels, self.crs_transformer, self.class_map)
            geojson_str = json.dumps(geojson)
            str_to_file(geojson_str, self.uri)
        else:
            raise ValueError('Cannot save with writable=False')
