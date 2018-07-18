from rastervision.core.label_store import LabelStore
from rastervision.labels.object_detection_labels import (
    ObjectDetectionLabels)


class ObjectDetectionLabelStore(LabelStore):
    def __init__(self):
        self.clear()

    def clear(self):
        self.labels = ObjectDetectionLabels.make_empty()

    def set_labels(self, labels):
        self.labels = labels

    def get_labels(self, window=None):
        if window is None:
            return self.labels

        return ObjectDetectionLabels.get_overlapping(self.labels, window)

    def extend(self, labels):
        self.labels = ObjectDetectionLabels.concatenate(
            self.labels, labels)

    def save(self):
        raise NotImplementedError()
