from typing import TYPE_CHECKING, Any, Optional, Tuple

from rastervision.core.data.utils import match_bboxes

if TYPE_CHECKING:
    from rastervision.core.box import Box
    from rastervision.core.data import (RasterSource, LabelSource, LabelStore)


class Scene:
    """The raster data and labels associated with an area of interest."""

    def __init__(self,
                 id: str,
                 raster_source: 'RasterSource',
                 label_source: Optional['LabelSource'] = None,
                 label_store: Optional['LabelStore'] = None,
                 aoi_polygons: Optional[list] = None):
        """Constructor.

        During initialization, ``Scene`` attempts to set the extents of the
        given ``label_source`` and the ``label_store`` to be identical to the
        extent of the given ``raster_source``.

        Args:
            id: ID for this scene.
            raster_source: Source of imagery for this scene.
            label_source: Source of labels for this scene.
            label_store: Store of predictions for this scene.
            aoi: Optional list of AOI polygons in pixel coordinates.
        """
        if label_source is not None:
            match_bboxes(raster_source, label_source)

        if label_store is not None:
            match_bboxes(raster_source, label_store)

        self.id = id
        self.raster_source = raster_source
        self.label_source = label_source
        self.label_store = label_store

        if aoi_polygons is None:
            self.aoi_polygons = []
        else:
            self.aoi_polygons = aoi_polygons

    @property
    def extent(self) -> 'Box':
        """Extent of the associated :class:`.RasterSource`."""
        return self.raster_source.extent

    def __getitem__(self, key: Any) -> Tuple[Any, Any]:
        x = self.raster_source[key]
        if self.label_source is not None:
            y = self.label_source[key]
        else:
            y = None
        return x, y
