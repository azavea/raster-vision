class Scene():
    """The raster data and labels associated with an area of interest."""

    def __init__(self,
                 id=None,
                 raster_source=None,
                 ground_truth_label_store=None,
                 prediction_label_store=None):
        """Construct a new Scene.

        Args:
            id: optional string
            raster_source: optional RasterSource
            ground_truth_label_store: optional LabelStore
            prediction_label_store: optional LabelStore
        """
        self.id = id
        self.raster_source = raster_source
        self.ground_truth_label_store = ground_truth_label_store
        self.prediction_label_store = prediction_label_store
