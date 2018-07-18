import numpy as np

from rastervision.core.labels import Labels
from rastervision.core.box import Box


# TODO also store scores
class ClassificationLabels(Labels):
    def __init__(self):
        self.cell_to_class_id = {}

    """Represents a spatial grid of cells associated with classes."""
    def set_cell(self, cell, class_id):
        """Set cell and its class_id.

        Args:
            cell: (Box)
        """
        self.cell_to_class_id[cell.tuple_format()] = class_id

    def get_cell_class_id(self, cell):
        return self.cell_to_class_id.get(cell.tuple_format())

    def get_singleton_labels(self, cell):
        class_id = self.get_cell_class_id(cell)
        labels = ClassificationLabels()
        labels.set_cell(cell, class_id)
        return labels

    def get_cells(self):
        return [Box.from_npbox(box_tup)
                for box_tup in self.cell_to_class_id.keys()]

    def get_class_ids(self):
        return list(self.cell_to_class_id.values())

    def extend(self, labels):
        for cell in labels.get_cells():
            class_id = labels.get_cell_class_id(cell)
            self.set_cell(cell, class_id)
