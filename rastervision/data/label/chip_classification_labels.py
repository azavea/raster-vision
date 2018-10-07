from rastervision.core.box import Box
from rastervision.data.label import Labels


class ChipClassificationLabels(Labels):
    """Represents a spatial grid of cells associated with classes."""

    def __init__(self):
        self.cell_to_class_id = {}

    def __len__(self):
        return len(self.cell_to_class_id)

    def __eq__(self, other):
        return (isinstance(other, ChipClassificationLabels)
                and self.cell_to_class_id == other.cell_to_class_id)

    def __add__(self, other):
        result = ChipClassificationLabels()
        result.extend(self)
        result.extend(other)
        return result

    def filter_by_aoi(self, aoi_polygons):
        result = ChipClassificationLabels()
        for cell in self.cell_to_class_id:
            cell_box = Box.from_tuple(cell)
            cell_poly = cell_box.to_shapely()
            for aoi in aoi_polygons:
                if cell_poly.within(aoi):
                    (class_id, scores) = self.cell_to_class_id[cell]
                    result.set_cell(cell_box, class_id, scores)
        return result

    def set_cell(self, cell, class_id, scores=None):
        """Set cell and its class_id.

        Args:
            cell: (Box)
            class_id: int
            scores: 1d numpy array of probabilities for each class
        """
        if scores is not None:
            scores = list(map(lambda x: float(x), list(scores)))
        self.cell_to_class_id[cell.tuple_format()] = (class_id, scores)

    def get_cell_class_id(self, cell):
        """Return class_id for a cell.

        Args:
            cell: (Box)
        """
        result = self.cell_to_class_id.get(cell.tuple_format())
        if result:
            return result[0]
        else:
            return None

    def get_cell_scores(self, cell):
        """Return scores for a cell.

        Args:
            cell: (Box)
        """
        result = self.cell_to_class_id.get(cell.tuple_format())
        if result:
            return result[1]
        else:
            return None

    def get_cell_values(self, cell):
        """Return a tuple of (class_id, scores) for a cell.

        Args:
            cell: (Box)
        """
        return self.cell_to_class_id.get(cell.tuple_format())

    def get_singleton_labels(self, cell):
        """Return Labels object representing a single cell.

        Args:
            cell: (Box)
        """
        class_id, scores = self.get_cell_values(cell)
        labels = ChipClassificationLabels()
        labels.set_cell(cell, class_id, scores)
        return labels

    def get_cells(self):
        """Return list of all cells (list of Box)."""
        return [
            Box.from_npbox(box_tup)
            for box_tup in self.cell_to_class_id.keys()
        ]

    def get_class_ids(self):
        """Return list of class_ids for all cells."""
        return list(map(lambda x: x[0], self.cell_to_class_id.values()))

    def get_scores(self):
        """Return list of scores for all cells."""
        return list(map(lambda x: x[1], self.cell_to_class_id.values()))

    def get_values(self):
        """Return list of class_ids and scores for all cells."""
        return list(self.cell_to_class_id.values())

    def extend(self, labels):
        """Adds cells contained in labels.

        Args:
            labels: ChipClassificationLabels
        """
        for cell in labels.get_cells():
            class_id, scores = labels.get_cell_values(cell)
            self.set_cell(cell, class_id, scores)
