import numpy as np

from rastervision.core.labels import Labels
from rastervision.core.box import Box


class ClassificationLabels(Labels):
    """Represents a spatial grid of cells associated with classes."""
    def __init__(self):
        # Mapping from Box tuple to int. This is a sparse representation of
        # the grid.
        self.cell_to_class_id = {}

    def get_class_id(self):
        """Get the class_id of the lone cell.

        Raises ValueError when there is more than one cell.
        """
        if len(self.cell_to_class_id) != 1:
            raise ValueError(
                'Needs to represent a single cell to get the class_id')
        else:
            return list(self.cell_to_class_id.values())[0]

    def set_cell(self, cell, class_id):
        """Set cell and its class_id.

        Args:
            cell: (Box)
        """
        self.cell_to_class_id[cell.tuple_format()] = class_id

    def get_cell_class_id(self, cell):
        return self.cell_to_class_id.get(cell.tuple_format())

    def get_cells(self):
        return [Box.from_npbox(box_tup)
                for box_tup in self.cell_to_class_id.keys()]

    def get_class_ids(self):
        return list(self.cell_to_class_id.values())
