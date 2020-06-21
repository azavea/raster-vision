from rastervision.core.data.label import ObjectDetectionLabels
from rastervision.core.data.label_source import LabelSource


class ObjectDetectionLabelSource(LabelSource):
    """A read-only label source for object detection."""

    def __init__(self, vector_source, extent=None):
        """Constructor.

        Args:
            vector_source: (VectorSource)
            extent: Box used to filter the labels by extent
        """
        self.labels = ObjectDetectionLabels.from_geojson(
            vector_source.get_geojson(), extent=extent)

    def get_labels(self, window=None):
        if window is None:
            return self.labels

        return ObjectDetectionLabels.get_overlapping(self.labels, window)
