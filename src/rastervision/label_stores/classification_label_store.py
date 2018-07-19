from rastervision.core.label_store import LabelStore
from rastervision.labels.classification_labels import ClassificationLabels


class ClassificationLabelStore(LabelStore):
    def __init__(self):
        self.clear()

    def clear(self):
        self.labels = ClassificationLabels()

    def get_labels(self, window=None):
        """Get labels.

        Args:
            window: Box

        Returns:
            All labels if window is None. Otherwise, returns singleton labels
            where lone cell is equal to window.
        """
        if window is None:
            return self.labels
        return self.labels.get_singleton_labels(window)

    def set_labels(self, labels):
        self.labels = labels

    def extend(self, labels):
        self.labels.extend(labels)

    def save(self):
        raise NotImplementedError()
