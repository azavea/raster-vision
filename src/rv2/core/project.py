class Project():
    """The raster data and labels associated with an area of interest."""

    def __init__(self, raster_source=None,
                ground_truth_label_source=None,
                prediction_label_source=None):
        """Construct a new Project.

        Args:
            raster_source: optional RasterSource
            ground_truth_label_source: optional LabelSource
            prediction_label_source: optional LabelSource
        """
        self.raster_source = raster_source
        self.ground_truth_label_source = ground_truth_label_source
        self.prediction_label_source = prediction_label_source
