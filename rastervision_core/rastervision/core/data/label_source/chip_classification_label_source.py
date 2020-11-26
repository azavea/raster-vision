import numpy as np
from shapely.strtree import STRtree
from shapely.geometry import shape
from typing import TYPE_CHECKING, Iterable, Optional
import click

from rastervision.core.data.label import ChipClassificationLabels
from rastervision.core.data.label_source.label_source import LabelSource
from rastervision.core.box import Box

if TYPE_CHECKING:
    from rastervision.core.data.vector_source import VectorSource  # noqa
    from rastervision.core.data.class_config import ClassConfig  # noqa
    from rastervision.core.data.crs_transformer import CRSTransformer  # noqa
    from rastervision.core.data.label_source.chip_classification_label_source_config import (  # noqa
        ChipClassificationLabelSourceConfig)  # noqa


def infer_cell(cell, str_tree, ioa_thresh, use_intersection_over_cell,
               background_class_id, pick_min_class_id) -> int:
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
        class_id = background_class_id
    elif pick_min_class_id:
        class_id = min(class_ids)
    else:
        # Pick class_id of the polygon with the biggest intersection over
        # cell. If there is a tie, pick the first.
        class_id = class_ids[np.argmax(inter_over_cells)]

    return class_id


def infer_labels(cells, str_tree, ioa_thresh, use_intersection_over_cell,
                 pick_min_class_id,
                 background_class_id) -> ChipClassificationLabels:
    """Infer ChipClassificationLabels grid from GeoJSON containing polygons.

    Given GeoJSON with polygons associated with class_ids, infer a grid of
    cells and class_ids that best captures the contents of each cell. See infer_cell for
    info on the args.

    Args:
        geojson: dict in normalized GeoJSON format (see VectorSource)

    Returns:
        ChipClassificationLabels
    """
    labels = ChipClassificationLabels()

    with click.progressbar(cells, label='Inferring labels') as bar:
        for cell in bar:
            class_id = infer_cell(cell, str_tree, ioa_thresh,
                                  use_intersection_over_cell,
                                  background_class_id, pick_min_class_id)
            labels.set_cell(cell, class_id)
    return labels


def read_labels(geojson, extent=None) -> ChipClassificationLabels:
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


def make_str_tree(geojson):
    # We need to associate class_id with each geom. Monkey-patching it onto the geom
    # seems like a bad idea, but it's the only straightforward way of doing this
    # that I've been able to find.
    geoms = []
    for f in geojson['features']:
        g = shape(f['geometry'])
        g.class_id = f['properties']['class_id']
        geoms.append(g)
    str_tree = STRtree(geoms)
    return str_tree


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
                 label_source_config: 'ChipClassificationLabelSourceConfig',
                 vector_source: 'VectorSource',
                 class_config: 'ClassConfig',
                 crs_transformer: 'CRSTransformer',
                 extent: Optional[Box] = None,
                 lazy: bool = False):
        """Constructs a LabelSource for chip classification.

        Args:
            extent (box, optional): Box used to filter the labels by extent or
                compute grid. This is only needed if infer_cells is True or if
                it is False and you want to filter cells by extent. Defaults to
                None.
            lazy (bool, optional): If True, labels are not populated during
                initialization. Defaults to False.
        """
        self.cfg = label_source_config
        self.geojson = vector_source.get_geojson()
        self.extent = extent
        self.str_tree = None

        self.labels = ChipClassificationLabels()
        if not lazy:
            self.populate_labels()

    def populate_labels(self, cells: Optional[Iterable[Box]] = None) -> None:
        """Populate self.labels by either reading or inferring.

        If cfg.infer_cells is True or specific cells are given, the labels are
        inferred. Otherwise, they are read from the geojson.
        """
        if self.cfg.infer_cells or cells is not None:
            self.labels = self.infer_cells(cells=cells)
        else:
            self.labels = read_labels(self.geojson, extent=self.extent)

    def infer_cells(self, cells: Optional[Iterable[Box]] = None
                    ) -> ChipClassificationLabels:
        """Infer labels for a list of cells. Only cells whose labels are not
        already known are inferred.

        Args:
            cells (Optional[Iterable[Box]], optional): Cells whose labels are
                to be inferred. Defaults to None.

        Returns:
            ChipClassificationLabels: labels
        """
        cfg = self.cfg
        if cells is None:
            cells = self.extent.get_windows(cfg.cell_sz, cfg.cell_sz)

        known_cells = [c for c in cells if c in self.labels]
        unknown_cells = [c for c in cells if c not in self.labels]

        if self.str_tree is None:
            self.str_tree = make_str_tree(self.geojson)

        labels = infer_labels(
            cells=unknown_cells,
            str_tree=self.str_tree,
            ioa_thresh=cfg.ioa_thresh,
            use_intersection_over_cell=cfg.use_intersection_over_cell,
            pick_min_class_id=cfg.pick_min_class_id,
            background_class_id=cfg.background_class_id)

        for cell in known_cells:
            class_id = self.labels.get_cell_class_id(cell)
            labels.set_cell(cell, class_id)

        return labels

    def infer_cell(self, cell: Optional[Box] = None) -> int:
        """Infer and return the label for a single cell."""
        cfg = self.cfg

        if self.str_tree is None:
            self.str_tree = make_str_tree(self.geojson)

        label = infer_cell(
            cell=cell,
            str_tree=self.str_tree,
            ioa_thresh=cfg.ioa_thresh,
            use_intersection_over_cell=cfg.use_intersection_over_cell,
            pick_min_class_id=cfg.pick_min_class_id,
            background_class_id=cfg.background_class_id)
        return label

    def get_labels(self,
                   window: Optional[Box] = None) -> ChipClassificationLabels:
        if window is None:
            return self.labels
        return self.labels.get_singleton_labels(window)

    def __getitem__(self, window: Box) -> int:
        """Return label for a window, inferring it if it is not already known.
        """
        if window in self.labels:
            return self.labels.get_cell_class_id(window)
        label = self.infer_cell(cell=window)
        self.labels.set_cell(window, label)
        return label
