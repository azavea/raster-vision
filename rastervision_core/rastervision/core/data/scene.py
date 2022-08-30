from typing import TYPE_CHECKING, Any, Optional, Tuple

from rastervision.core.box import Box

if TYPE_CHECKING:
    from rastervision.core.data import (RasterSource, LabelSource, LabelStore)


class Scene:
    """The raster data and labels associated with an area of interest."""

    def __init__(self,
                 id: str,
                 raster_source: 'RasterSource',
                 label_source: Optional['LabelSource'] = None,
                 label_store: Optional['LabelStore'] = None,
                 aoi_polygons: Optional[list] = None):
        """Construct a new Scene.

        Args:
            id: ID for this scene
            raster_source: RasterSource for this scene
            ground_truth_label_store: optional LabelSource
            label_store: optional LabelStore
            aoi: Optional list of AOI polygons in pixel coordinates
        """
        self.id = id
        self.raster_source = raster_source
        self.label_source = label_source
        self.label_store = label_store
        if aoi_polygons is None:
            self.aoi_polygons = []
        else:
            self.aoi_polygons = aoi_polygons

    def __getitem__(self, window: Box) -> Tuple[Any, Any]:
        x = self.raster_source[window]
        if self.label_source is not None:
            y = self.label_source[window]
        else:
            y = None
        return x, y
