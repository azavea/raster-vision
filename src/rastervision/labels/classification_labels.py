import numpy as np

from rastervision.core.labels import Labels
from rastervision.core.box import Box


# TODO also store scores
class ClassificationLabels(Labels):
    """Represents a spatial grid of cells associated with classes."""

    def __init__(self):
        self.cell_to_class_id = {}

    def __len__(self):
        return len(self.cell_to_class_id)

    def set_cell(self, cell, class_id):
        """Set cell and its class_id.

        Args:
            cell: (Box)
            class_id: int
        """
        self.cell_to_class_id[cell.tuple_format()] = class_id

    def get_cell_class_id(self, cell):
        """Return class_id for a cell.

        Args:
            cell: (Box)
        """
        return self.cell_to_class_id.get(cell.tuple_format())

    def get_singleton_labels(self, cell):
        """Return Labels object representing a single cell.

        Args:
            cell: (Box)
        """
        class_id = self.get_cell_class_id(cell)
        labels = ClassificationLabels()
        labels.set_cell(cell, class_id)
        return labels

    def get_cells(self):
        """Return list of all cells (list of Box)."""
        return [
            Box.from_npbox(box_tup)
            for box_tup in self.cell_to_class_id.keys()
        ]

    def get_class_ids(self):
        """Return list of class_ids for all cells."""
        return list(self.cell_to_class_id.values())

    def extend(self, labels):
        """Adds cells contained in labels.

        Args:
            labels: ClassificationLabels
        """
        for cell in labels.get_cells():
            class_id = labels.get_cell_class_id(cell)
            self.set_cell(cell, class_id)
