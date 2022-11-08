from typing import TYPE_CHECKING, Any, Iterable, List, Optional

import numpy as np
import geopandas as gpd

from rastervision.core.data.label import ChipClassificationLabels
from rastervision.core.data.label_source.label_source import LabelSource
from rastervision.core.box import Box

if TYPE_CHECKING:
    from rastervision.core.data import (ChipClassificationLabelSourceConfig,
                                        VectorSource)


def infer_cells(cells: List[Box], labels_df: gpd.GeoDataFrame,
                ioa_thresh: float, use_intersection_over_cell: bool,
                pick_min_class_id: bool,
                background_class_id: int) -> ChipClassificationLabels:
    """Infer ChipClassificationLabels grid from GeoJSON containing polygons.

    Given GeoJSON with polygons associated with class_ids, infer a grid of
    cells and class_ids that best captures the contents of each cell.

    For each cell, the problem is to infer the class_id that best captures the
    content of the cell. This is non-trivial since there can be multiple
    polygons of differing classes overlapping with the cell. Any polygons that
    sufficiently overlaps with the cell are in the running for setting the
    class_id. If there are none in the running, the cell is either
    considered null or background.

    Args:
        ioa_thresh: (float) the minimum IOA of a polygon and cell for that
            polygon to be a candidate for setting the class_id
        use_intersection_over_cell: (bool) If true, then use the area of the
            cell as the denominator in the IOA. Otherwise, use the area of the
            polygon.
        background_class_id: (None or int) If not None, class_id to use as the
            background class; ie. the one that is used when a window contains
            no boxes.
        pick_min_class_id: If true, the class_id for a cell is the minimum
            class_id of the boxes in that cell. Otherwise, pick the class_id of
            the box covering the greatest area.

    Returns:
        ChipClassificationLabels
    """
    cells_df = gpd.GeoDataFrame(
        data={'cell_id': range(len(cells))},
        geometry=[c.to_shapely() for c in cells])

    # duplicate geometry columns so that they are retained after the join
    cells_df.loc[:, 'geometry_cell'] = cells_df.geometry
    labels_df.loc[:, 'geometry_label'] = labels_df.geometry

    # Left-join cells to label polygons based on intersection. The result is a
    # table with each cell matched to all polygons that intersect it; i.e.,
    # there will be a row for each unique (cell, polygon) combination. Cells
    # that didn't match any labels will have missing values as their class_ids.
    df: gpd.GeoDataFrame = cells_df.sjoin(
        labels_df, how='left', predicate='intersects')
    df.loc[:, 'geometry_intersection'] = df['geometry_cell'].intersection(
        df['geometry_label'])

    if use_intersection_over_cell:
        ioa = (df['geometry_intersection'].area / df['geometry_cell'].area)
    else:
        # intersection over label-polygon
        ioa = (df['geometry_intersection'].area / df['geometry_label'].area)
    df.loc[:, 'ioa'] = ioa.fillna(-1)

    # labels with IOA below threshold cannot contribute their class_id
    df.loc[df['ioa'] < ioa_thresh, 'class_id'] = None

    # Assign background_class_id to cells /wo a class_id. This includes both
    # unmatched cells and ones whose ioa fell below the ioa_thresh.
    df.loc[df['class_id'].isna(), 'class_id'] = background_class_id

    # break ties (i.e. one cell matched to multiple label polygons)
    if pick_min_class_id:
        df = df.sort_values('class_id').drop_duplicates(
            ['cell_id'], keep='first')
    else:
        # largest IOA
        df = df.sort_values('ioa').drop_duplicates(['cell_id'], keep='last')

    boxes = [Box.from_shapely(c).to_int() for c in df['geometry_cell']]
    class_ids = df['class_id'].astype(int)
    cells_to_class_id = {
        cell: (class_id, None)
        for cell, class_id in zip(boxes, class_ids)
    }
    labels = ChipClassificationLabels(cells_to_class_id)
    return labels


def read_labels(labels_df: gpd.GeoDataFrame,
                extent: Optional[Box] = None) -> ChipClassificationLabels:
    """Convert GeoDataFrame to ChipClassificationLabels.

    If the GeoDataFrame already contains a grid of cells, then
    ChipClassificationLabels can be constructed in a straightforward manner
    without having to infer the class of cells.

    If extent is given, only labels that intersect with it are returned.

    Args:
        geojson: dict in normalized GeoJSON format (see VectorSource)
        extent: Box in pixel coords

    Returns:
       ChipClassificationLabels
    """
    if extent is not None:
        extent_polygon = extent.to_shapely()
        labels_df = labels_df[labels_df.intersects(extent_polygon)]
        boxes = np.array([
            Box.from_shapely(c).to_int().shift_origin(extent)
            for c in labels_df.geometry
        ])
    else:
        boxes = np.array(
            [Box.from_shapely(c).to_int() for c in labels_df.geometry])
    class_ids = labels_df['class_id'].astype(int)
    cells_to_class_id = {
        cell: (class_id, None)
        for cell, class_id in zip(boxes, class_ids)
    }
    labels = ChipClassificationLabels(cells_to_class_id)
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
                 label_source_config: 'ChipClassificationLabelSourceConfig',
                 vector_source: 'VectorSource',
                 extent: Box = None,
                 lazy: bool = False):
        """Constructs a LabelSource for chip classification.

        Args:
            label_source_config (ChipClassificationLabelSourceConfig): Config
                for class inference.
            vector_source (VectorSource): Source of vector labels.
            extent (Box): Box used to filter the labels by extent or
                compute grid.
            lazy (bool, optional): If True, labels are not populated during
                initialization. Defaults to False.
        """
        self.cfg = label_source_config
        self._extent = extent
        self.labels_df = vector_source.get_dataframe()
        self.validate_labels(self.labels_df)

        self.labels = ChipClassificationLabels.make_empty()
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
            self.labels = read_labels(self.labels_df, extent=self.extent)

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
            if cfg.cell_sz is None:
                raise ValueError('cell_sz is not set.')
            cells = self.extent.get_windows(cfg.cell_sz, cfg.cell_sz)
        else:
            cells = [cell.shift_origin(self.extent) for cell in cells]

        known_cells = [c for c in cells if c in self.labels]
        unknown_cells = [c for c in cells if c not in self.labels]

        labels = infer_cells(
            cells=unknown_cells,
            labels_df=self.labels_df,
            ioa_thresh=cfg.ioa_thresh,
            use_intersection_over_cell=cfg.use_intersection_over_cell,
            pick_min_class_id=cfg.pick_min_class_id,
            background_class_id=cfg.background_class_id)

        for cell in known_cells:
            class_id = self.labels.get_cell_class_id(cell)
            labels.set_cell(cell, class_id)

        return labels

    def get_labels(self,
                   window: Optional[Box] = None) -> ChipClassificationLabels:
        if window is None:
            return self.labels
        window = window.shift_origin(self.extent)
        return self.labels.get_singleton_labels(window)

    def __getitem__(self, key: Any) -> int:
        """Return label for a window, inferring it if it is not already known.
        """
        if isinstance(key, Box):
            window = key
            if window not in self.labels:
                self.labels += self.infer_cells(cells=[window])
            return self.labels[window].class_id
        else:
            return super().__getitem__(key)

    def validate_labels(self, df: gpd.GeoDataFrame) -> None:
        geom_types = set(df.geom_type)
        if 'Point' in geom_types or 'LineString' in geom_types:
            raise ValueError(
                'LineStrings and Points are not supported '
                'in ChipClassificationLabelSource. Use BufferTransformer '
                'to buffer them into Polygons.')

        if 'class_id' not in df.columns:
            raise ValueError('All label polygons must have a class_id.')

    @property
    def extent(self) -> Box:
        return self._extent
