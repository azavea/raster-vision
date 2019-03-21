import numpy as np
from shapely.strtree import STRtree
from shapely.geometry import shape

import rastervision as rv
from rastervision.data import ChipClassificationLabels
from rastervision.data.label_source import LabelSource
from rastervision.core import Box


def infer_cell(cell, str_tree, ioa_thresh, use_intersection_over_cell,
               background_class_id, pick_min_class_id):
    """Infer the class_id of a cell given a set of polygons.

    Given a cell and a set of polygons, the problem is to infer the class_id
    that best captures the content of the cell. This is non-trivial since there
    can be multiple polygons of differing classes overlapping with the cell.
    Any polygons that sufficiently overlaps with the cell are in the running for
    setting the class_id. If there are none in the running, the cell is either
    considered null or background. See args for more details.

    Args:
        cell: Box
        str_tree: (STRtree) collection of geoms in scene used for geometric queries.
            The geoms need to have class_id monkey-patched onto them.
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
    cell_geom = cell.to_shapely()
    inter_polys = str_tree.query(cell_geom)

    inter_over_cells = []
    inter_over_polys = []
    class_ids = []

    # Find polygons whose intersection with the cell is big enough.
    for poly in inter_polys:
        inter = poly.intersection(cell_geom)
        inter_over_cell = inter.area / cell_geom.area
        inter_over_poly = inter.area / poly.area

        if use_intersection_over_cell:
            enough_inter = inter_over_cell >= ioa_thresh
        else:
            enough_inter = inter_over_poly >= ioa_thresh

        if enough_inter:
            inter_over_cells.append(inter_over_cell)
            inter_over_polys.append(inter_over_poly)
            class_ids.append(poly.class_id)

    # Infer class id for cell.
    if len(class_ids) == 0:
        class_id = (None if background_class_id == 0 else background_class_id)
    elif pick_min_class_id:
        class_id = min(class_ids)
    else:
        # Pick class_id of the polygon with the biggest intersection over
        # cell. If there is a tie, pick the first.
        class_id = class_ids[np.argmax(inter_over_cells)]

    return class_id


def infer_labels(geojson, extent, cell_size, ioa_thresh,
                 use_intersection_over_cell, pick_min_class_id,
                 background_class_id):
    """Infer ChipClassificationLabels grid from GeoJSON containing polygons.

    Given GeoJSON with polygons associated with class_ids, infer a grid of
    cells and class_ids that best captures the contents of each cell. See infer_cell for
    info on the args.

    Args:
        geojson: dict in normalized GeoJSON format (see VectorSource)
        extent: Box representing the bounds of the grid

    Returns:
        ChipClassificationLabels
    """
    labels = ChipClassificationLabels()
    cells = extent.get_windows(cell_size, cell_size)

    # We need to associate class_id with each geom. Monkey-patching it onto the geom
    # seems like a bad idea, but it's the only straightforward way of doing this
    # that I've been able to find.
    geoms = []
    for f in geojson['features']:
        g = shape(f['geometry'])
        g.class_id = f['properties']['class_id']
        geoms.append(g)
    str_tree = STRtree(geoms)

    for cell in cells:
        class_id = infer_cell(cell, str_tree, ioa_thresh,
                              use_intersection_over_cell, background_class_id,
                              pick_min_class_id)
        labels.set_cell(cell, class_id)
    return labels


def read_labels(geojson, extent=None):
    """Convert GeoJSON to ChipClassificationLabels.

    If the GeoJSON already contains a grid of cells, then it can be constructed
    in a straightforward manner without having to infer the class of cells.

    If extent is given, only labels that intersect with the extent are returned.

    Args:
        geojson: dict in normalized GeoJSON format (see VectorSource)
        extent: Box in pixel coords

    Returns:
       ChipClassificationLabels
    """
    labels = ChipClassificationLabels()

    for f in geojson['features']:
        geom = shape(f['geometry'])
        (xmin, ymin, xmax, ymax) = geom.bounds
        cell = Box(ymin, xmin, ymax, xmax)
        if extent is not None and not cell.to_shapely().intersects(
                extent.to_shapely()):
            continue

        props = f['properties']
        class_id = props['class_id']
        scores = props.get('scores')
        labels.set_cell(cell, class_id, scores)

    return labels


class ChipClassificationLabelSource(LabelSource):
    """A source of chip classification labels.

    Ideally the vector_source contains a square for each cell in the grid. But
    in reality, it can be difficult to label imagery in such an exhaustive way.
    So, this can also handle sources with non-overlapping polygons that
    do not necessarily cover the entire extent. It infers the grid of cells
    and associated class_ids using the extent and options if infer_cells is
    set to True.
    """

    def __init__(self,
                 vector_source,
                 crs_transformer,
                 class_map,
                 extent=None,
                 ioa_thresh=None,
                 use_intersection_over_cell=False,
                 pick_min_class_id=False,
                 background_class_id=None,
                 cell_size=None,
                 infer_cells=False):
        """Constructs a LabelSource for ChipClassificaiton backed by a GeoJSON file.

        Args:
            vector_source: (VectorSource or str)
            crs_transformer: CRSTransformer to convert from map coords in label
                in GeoJSON file to pixel coords.
            class_map: ClassMap used to infer class_ids from class_name
                (or label) field
            extent: Box used to filter the labels by extent or compute grid
        """
        if isinstance(vector_source, str):
            provider = rv._registry.get_vector_source_default_provider(
                vector_source)
            vector_source = provider.construct(vector_source) \
                .create_source(
                    crs_transformer=crs_transformer, extent=extent, class_map=class_map)

        geojson = vector_source.get_geojson()

        if infer_cells:
            self.labels = infer_labels(geojson, extent, cell_size, ioa_thresh,
                                       use_intersection_over_cell,
                                       pick_min_class_id, background_class_id)
        else:
            self.labels = read_labels(geojson, extent)

    def get_labels(self, window=None):
        if window is None:
            return self.labels
        return self.labels.get_singleton_labels(window)
