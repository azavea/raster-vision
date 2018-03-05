class Project():
    def __init__(self, raster_source=None,
                ground_truth_annotation_source=None,
                prediction_annotation_source=None):
        self.raster_source = raster_source
        self.ground_truth_annotation_source = ground_truth_annotation_source
        self.prediction_annotation_source = prediction_annotation_source
