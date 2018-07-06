from rastervision.core.label_store import LabelStore
from rastervision.labels.classification_labels import ClassificationLabels


class ClassificationLabelStore(LabelStore):
    """Represents source of a spatial grid of cells associated with classes."""

    def __init__(self, extent, cell_size):
        self.clear()
        self.set_grid(extent, cell_size)

    def set_grid(self, extent, cell_size):
        """Set parameters implicitly defining the spatial grid."""
        self.extent = extent
        self.cell_size = cell_size

    def is_valid_cell(self, box):
        """Returns True if box corresponds to a grid cell."""
        return (box.ymin % self.cell_size == 0 and
                box.ymin < self.extent.get_height() and
                box.xmin % self.cell_size == 0 and
                box.xmin < self.extent.get_width())

    def get_labels(self, window):
        """Get the labels for a window.

        Args:
            window: (Box)

        Returns:
            ClassificationLabels object containing a single cell.

        Raises:
            ValueError: window is not valid grid cell.
        """
        if not self.is_valid_cell(window):
            raise ValueError('window needs to be valid grid cell')
        class_id = self.labels.get_cell_class_id(window)
        labels = ClassificationLabels()
        labels.set_cell(window, class_id)
        return labels

    def get_all_labels(self):
        return self.labels

    def set_cell(self, window, class_id):
        self.labels.set_cell(window, class_id)

    def extend(self, window, labels):
        self.set_cell(window, labels.get_class_id())

    def post_process(self, options):
        pass

    def clear(self):
        self.labels = ClassificationLabels()

    def save(self):
        pass
