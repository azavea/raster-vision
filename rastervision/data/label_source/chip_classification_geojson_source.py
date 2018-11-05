import numpy as np
from shapely.strtree import STRtree
from shapely import geometry

from rastervision.data import ChipClassificationLabels
from rastervision.data.label_source import LabelSource
from rastervision.data.label_source.utils import (
    add_classes_to_geojson, load_label_store_json,
    geojson_to_chip_classification_labels)
from rastervision.data.utils import geojson_to_shapes


def infer_cell(shapes, cell, ioa_thresh, use_intersection_over_cell,
               background_class_id, pick_min_class_id):
    """Infer the class_id of a cell given a set of polygons.

    Given a cell and a set of polygons, the problem is to infer the class_id
    that best captures the content of the cell. This is non-trivial since there
    can be multiple polygons of differing classes overlapping with the cell.
    Any polygons that sufficiently overlap with the cell are in the running for
    setting the class_id. If there are none in the running, the cell is either
    considered null or background. See args for more details.

    Args:
        shapes: List of (shapely.geometry, class_id) tuples
        cell: Box
        ioa_thresh: (float) the minimum IOA of a polygon and cell for that
            polygon to be a candidate for setting the class_id
        use_intersection_over_cell: (bool) If true, then use the area of the
            cell as the denominator in the IOA. Otherwise, use the area of the
            polygon.
        background_class_id: (None or int) If not None, class_id to use as the
            background class; ie. the one that is used when a window contains
            no boxes. If not set, empty windows have None set as their class_id
            which is considered a null value.
        pick_min_class_id: If true, the class_id for a cell is the minimum
            class_id of the boxes in that cell. Otherwise, pick the class_id of
            the box covering the greatest area.
    """
    str_tree = STRtree([shape for shape, class_id in shapes])
    # Monkey-patching class_id onto shapely.geom is not a good idea because
    # if you transform it, the class_id will be lost, but this works here. I wanted to
    # use a dictionary to associate shape with class_id, but couldn't because they are
    # mutable.
    for shape, class_id in shapes:
        shape.class_id = class_id

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
        class_id = (None if background_class_id == 0 else background_class_id)
    elif pick_min_class_id:
        class_id = min(class_ids)
    else:
        # Pick class_id of the polygon with the biggest intersection over
        # cell. If there is a tie, pick the first.
        class_id = class_ids[np.argmax(intersection_over_cells)]

    return class_id


def infer_labels(geojson, crs_transformer, extent, cell_size, ioa_thresh,
                 use_intersection_over_cell, pick_min_class_id,
                 background_class_id):
    """Infer ChipClassificationLabels grid from GeoJSON containing polygons.

    Given GeoJSON with polygons associated with class_ids, infer a grid of
    cells and class_ids that best captures the contents of each cell.

    Args:
        geojson: dict in GeoJSON format
        crs_transformer: CRSTransformer used to convert from map to pixel based
            coordinates
        extent: Box representing the bounds of the grid

    Returns:
        ChipClassificationLabels
    """
    shapes = geojson_to_shapes(geojson, crs_transformer)
    # Only keep polygons and multipolygons.
    # TODO: handle linestrings
    shapes = [(shape, class_id) for shape, class_id in shapes
              if type(shape) in [geometry.Polygon, geometry.MultiPolygon]]
    labels = ChipClassificationLabels()

    cells = extent.get_windows(cell_size, cell_size)
    for cell in cells:
        class_id = infer_cell(shapes, cell, ioa_thresh,
                              use_intersection_over_cell, background_class_id,
                              pick_min_class_id)
        labels.set_cell(cell, class_id)
    return labels


def read_labels(geojson, crs_transformer, extent=None):
    """Construct ChipClassificationLabels from GeoJSON containing grid of cells.

    If the GeoJSON already contains a grid of cells, then it can be constructed
    in a straightforward manner without having to infer the class of cells.

    Args:
        geojson: dict in GeoJSON format
        crs_transformer: CRSTransformer used to convert from map to pixel based
            coordinates
        extent: Box used to filter the grid in the geojson so the grid
            only contains cells that overlap with the extent

    Returns:
        ChipClassificationLabels
    """
    return geojson_to_chip_classification_labels(geojson, crs_transformer,
                                                 extent)


def load_geojson(geojson, crs_transformer, extent, infer_cells, cell_size,
                 ioa_thresh, use_intersection_over_cell, pick_min_class_id,
                 background_class_id):
    """Construct ChipClassificationLabels from GeoJSON.

    Either infers or reads the grid from the GeoJSON depending on the
    value of infer_cells.

    Args:
        geojson: dict in GeoJSON format
        crs_transformer: CRSTransformer used to convert from map to pixel based
            coordinates
        extent: Box representing the bounds of the grid
    Returns:
        ChipClassificationLabels
    """
    if infer_cells:
        labels = infer_labels(geojson, crs_transformer, extent, cell_size,
                              ioa_thresh, use_intersection_over_cell,
                              pick_min_class_id, background_class_id)
    else:
        labels = read_labels(geojson, crs_transformer, extent)

    return labels


class ChipClassificationGeoJSONSource(LabelSource):
    """A GeoJSON file with chip classification labels in it.

    Ideally the GeoJSON file contains a square for each cell in the grid. But
    in reality, it can be difficult to label imagery in such an exhaustive way.
    So, this can also handle GeoJSON files with non-overlapping polygons that
    do not necessarily cover the entire extent. It infers the grid of cells
    and associated class_ids using the extent and options if infer_cells is
    set to True.

    """

    def __init__(self,
                 uri,
                 crs_transformer,
                 class_map,
                 extent,
                 ioa_thresh=None,
                 use_intersection_over_cell=False,
                 pick_min_class_id=False,
                 background_class_id=None,
                 cell_size=None,
                 infer_cells=False):
        """Constructs a LabelSource for ChipClassificaiton backed by a GeoJSON file.

        Args:
            uri: uri of GeoJSON file containing labels
            crs_transformer: CRSTransformer to convert from map coords in label
                in GeoJSON file to pixel coords.
            class_map: ClassMap used to infer class_ids from class_name
                (or label) field
            extent: Box used to filter the labels by extent or compute grid
        """
        self.labels = ChipClassificationLabels()

        geojson = load_label_store_json(uri)
        if geojson:
            geojson = add_classes_to_geojson(geojson, class_map)
            self.labels = load_geojson(geojson, crs_transformer, extent,
                                       infer_cells, cell_size, ioa_thresh,
                                       use_intersection_over_cell,
                                       pick_min_class_id, background_class_id)

    def get_labels(self, window=None):
        if window is None:
            return self.labels
        return self.labels.get_singleton_labels(window)
