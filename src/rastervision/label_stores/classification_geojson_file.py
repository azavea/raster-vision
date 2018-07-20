import json

import numpy as np
from shapely.strtree import STRtree
from shapely import geometry

from rastervision.labels.classification_labels import (
    ClassificationLabels)
from rastervision.label_stores.object_detection_geojson_file import (
    geojson_to_labels as geojson_to_object_detection_labels)
from rastervision.label_stores.utils import (
    add_classes_to_geojson, load_label_store_json, boxes_to_geojson)
from rastervision.utils.files import str_to_file
from rastervision.label_stores.classification_label_store import (
        ClassificationLabelStore)


def get_str_tree(geojson_dict, crs_transformer):
    """Get shapely STRtree data structure for a set of polygons.

    Args:
        geojson_dict: dict in GeoJSON format with class_id property for each
            polygon
        crs_transformer: CRSTransformer used to convert from map to pixel
            coords

    Returns:
        shapely.strtree.STRtree
    """
    features = geojson_dict['features']
    json_polygons = []
    class_ids = []

    for feature in features:
        # Convert polygon to pixel coords.
        polygon = feature['geometry']['coordinates'][0]
        polygon = [crs_transformer.map_to_pixel(p) for p in polygon]
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


def infer_cell(str_tree, cell, ioa_thresh, use_intersection_over_cell,
               background_class_id, pick_min_class_id):
    """Infer the class_id of a cell given a set of polygons.

    Given a cell and a set of polygons, the problem is to infer the class_id
    that best captures the content of the cell. This is non-trivial since there
    can be multiple polygons of differing classes overlapping with the cell. Any
    polygons that sufficiently overlap with the cell are in the running for
    setting the class_id. If there are none in the running, the cell is either
    considered null or background. See args for more details.

    Args:
        str_tree: shapely.strtree.STRtree of polygons with class_id attributes
        cell: Box
        ioa_thresh: (float) the minimum IOA of a polygon and cell for that
            polygon to be a candidate for setting the class_id
        use_intersection_over_cell: (bool) If true, then use the area of the
            cell as the denominator in the IOA. Otherwise, use the area of the
            polygon.
        background_class_id: (None or int) If not None, class_id to use as the
            background class; ie. the one that is used when a window contains no
            boxes. If not set, empty windows have None set as their class_id
            which is considered a null value.
        pick_min_class_id: If true, the class_id for a cell is the minimum
            class_id of the boxes in that cell. Otherwise, pick the class_id of
            the box covering the greatest area.
    """
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

        if use_intersection_over_cell:
            enough_intersection = intersection_over_cell >= ioa_thresh
        else:
            enough_intersection = intersection_over_polygon >= ioa_thresh

        if enough_intersection:
            intersection_over_cells.append(intersection_over_cell)
            intersection_over_polygons.append(intersection_over_polygon)
            class_ids.append(polygon.class_id)

    # Infer class id for cell.
    if len(class_ids) == 0:
        class_id = (None if background_class_id == 0
                    else background_class_id)
    elif pick_min_class_id:
        class_id = min(class_ids)
    else:
        # Pick class_id of the polygon with the biggest intersection over
        # cell. If there is a tie, pick the first.
        class_id = class_ids[np.argmax(intersection_over_cells)]

    return class_id


def infer_labels(geojson_dict, crs_transformer, extent, options):
    """Infer ClassificationLabels grid from GeoJSON containing polygons.

    Given GeoJSON with polygons associated with class_ids, infer a grid of
    cells and class_ids that best captures the contents of each cell.

    Args:
        geojson_dict: dict in GeoJSON format
        crs_transformer: CRSTransformer used to convert from map to pixel based
            coordinates
        extent: Box representing the bounds of the grid
        options: rastervision.protos.label_store_pb2.ClassificationGeoJSONFile.Options

    Returns:
        ClassificationLabels
    """
    str_tree = get_str_tree(geojson_dict, crs_transformer)
    labels = ClassificationLabels()

    cells = extent.get_windows(options.cell_size, options.cell_size)
    for cell in cells:
        class_id = infer_cell(
            str_tree, cell, options.ioa_thresh,
            options.use_intersection_over_cell, options.background_class_id,
            options.pick_min_class_id)
        labels.set_cell(cell, class_id)
    return labels


def read_labels(geojson_dict, crs_transformer, extent):
    """Construct ClassificationLabels from GeoJSON containing grid of cells.

    If the GeoJSON already contains a grid of cells, then it can be constructed
    in a straightforward manner without having to infer the class of cells.

    Args:
        geojson_dict: dict in GeoJSON format
        crs_transformer: CRSTransformer used to convert from map to pixel based
            coordinates
        extent: Box used to filter the grid in the geojson_dict so the grid
            only contains cells that overlap with the extent

    Returns:
        ClassificationLabels
    """
    # Load as ObjectDetectionLabels and convert to ClassificationLabels.
    od_labels = geojson_to_object_detection_labels(
        geojson_dict, crs_transformer, extent)

    labels = ClassificationLabels()
    boxes = od_labels.get_boxes()
    class_ids = od_labels.get_class_ids()
    for box, class_id in zip(boxes, class_ids):
        labels.set_cell(box, class_id)

    return labels


def load_geojson(geojson_dict, crs_transformer, extent, options):
    """Construct ClassificationLabels from GeoJSON.

    Either infers or reads the grid from the GeoJSON depending on the
    value of options.infer_cells.

    Args:
        geojson_dict: dict in GeoJSON format
        crs_transformer: CRSTransformer used to convert from map to pixel based
            coordinates
        extent: Box representing the bounds of the grid
        options: rastervision.protos.label_store_pb2.ClassificationGeoJSONFile.Options
    Returns:
        ClassificationLabels
    """
    if options.infer_cells:
        labels = infer_labels(geojson_dict, crs_transformer, extent, options)
    else:
        labels = read_labels(geojson_dict, crs_transformer, extent)

    return labels


def to_geojson(labels, crs_transformer, class_map):
    """Convert ClassificationLabels to to GeoJSON.

    Outputs the grid in GeoJSON format as a list of squares each
    with a class_id and class_name property.

    Args:
        labels: ClassificationLabels
        crs_transformer: CRSTransformer used to convert from pixle to map based
            coordinates
        class_map: ClassMap used to infer class_name from class_ids in labels
    Returns:
        dict in GeoJSON format
    """
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
    def __init__(self, uri, crs_transformer, options, class_map,
                 extent, readable=True, writable=False):
        """Construct ClassificationLabelStore backed by a GeoJSON file.

        Args:
            uri: uri of GeoJSON file containing labels
            crs_transformer: CRSTransformer to convert from map coords in label
                in GeoJSON file to pixel coords.
            class_map: ClassMap used to infer class_ids from class_name
                (or label) field
            extent: Box used to filter the labels by extent or compute grid
            readable: if True, expect the file to exist
            writable: if True, allow writing to disk
        """
        self.uri = uri
        self.crs_transformer = crs_transformer
        self.class_map = class_map
        self.readable = readable
        self.writable = writable

        self.labels = ClassificationLabels()

        geojson_dict = load_label_store_json(uri, readable)
        if geojson_dict:
            geojson_dict = add_classes_to_geojson(geojson_dict, class_map)
            self.labels = load_geojson(
                geojson_dict, crs_transformer, extent, options)

    def save(self):
        """Save labels to URI if writable.

        Note that if the grid is inferred from polygons, only the grid will be
        written, not the original polygons.
        """
        if self.writable:
            geojson_dict = to_geojson(
                self.labels, self.crs_transformer, self.class_map)
            geojson_str = json.dumps(geojson_dict)
            str_to_file(geojson_str, self.uri)
        else:
            raise ValueError('Cannot save with writable=False')
