from typing import TYPE_CHECKING, Any, Optional, Tuple

from rastervision.core.data import ActivateMixin
from rastervision.core.box import Box

if TYPE_CHECKING:
    from rastervision.core.data import (RasterSource, LabelSource, LabelStore)


class Scene(ActivateMixin):
    """The raster data and labels associated with an area of interest."""

    def __init__(self,
                 id: str,
                 raster_source: 'RasterSource',
                 ground_truth_label_source: Optional['LabelSource'] = None,
                 prediction_label_store: Optional['LabelStore'] = None,
                 aoi_polygons: Optional[list] = None):
        """Construct a new Scene.

        Args:
            id: ID for this scene
            raster_source: RasterSource for this scene
            ground_truth_label_store: optional LabelSource
            prediction_label_store: optional LabelStore
            aoi: Optional list of AOI polygons
        """
        self.id = id
        self.raster_source = raster_source
        self.ground_truth_label_source = ground_truth_label_source
        self.prediction_label_store = prediction_label_store
        if aoi_polygons is None:
            self.aoi_polygons = []
        else:
            self.aoi_polygons = aoi_polygons

    @property
    def label_source(self) -> 'LabelSource':
        return self.ground_truth_label_source

    @property
    def label_store(self) -> 'LabelStore':
        return self.prediction_label_store

    def __getitem__(self, window: Box) -> Tuple[Any, Any]:
        x = self.raster_source[window]
        y = self.label_source[window]
        return x, y

    def _subcomponents_to_activate(self) -> list:
        return [
            self.raster_source, self.ground_truth_label_source,
            self.prediction_label_store
        ]

    def _activate(self):
        pass

    def _deactivate(self):
        pass
