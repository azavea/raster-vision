class Project():
    """The raster data and labels associated with an area of interest."""

    def __init__(self, raster_source=None,
                ground_truth_label_store=None,
                prediction_label_store=None):
        """Construct a new Project.

        Args:
            raster_source: optional RasterSource
            ground_truth_label_store: optional LabelStore
            prediction_label_store: optional LabelStore
        """
        self.raster_source = raster_source
        self.ground_truth_label_store = ground_truth_label_store
        self.prediction_label_store = prediction_label_store
