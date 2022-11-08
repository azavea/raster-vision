from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional,
                    Sequence, Tuple)
from dataclasses import dataclass

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data.label import Labels

if TYPE_CHECKING:
    from rastervision.core.data import (ClassConfig, CRSTransformer)
    from shapely.geometry import Polygon


@dataclass
class ClassificationLabel:
    class_id: int
    scores: Optional[Sequence[float]] = None

    def __iter__(self):
        return iter((self.class_id, self.scores))


class ChipClassificationLabels(Labels):
    """Represents a spatial grid of cells associated with classes."""

    def __init__(self,
                 cell_to_label: Optional[Dict[Box, Tuple[int, Optional[
                     Sequence[float]]]]] = None):
        if cell_to_label is None:
            cell_to_label = {}

        self.cell_to_label = {
            c: ClassificationLabel(*v)
            for c, v in cell_to_label.items()
        }

    def __len__(self) -> int:
        return len(self.cell_to_label)

    def __eq__(self, other: 'ChipClassificationLabels') -> bool:
        return (isinstance(other, ChipClassificationLabels)
                and self.cell_to_label == other.cell_to_label)

    def __add__(self, other: 'ChipClassificationLabels'
                ) -> 'ChipClassificationLabels':
        result = ChipClassificationLabels()
        result.extend(self)
        result.extend(other)
        return result

    def __contains__(self, cell: Box) -> bool:
        return cell in self.cell_to_label

    def __getitem__(self, cell: Box) -> ClassificationLabel:
        return self.cell_to_label[cell]

    def __setitem__(self, window: Box,
                    value: Tuple[int, Optional[Sequence[float]]]):
        class_id, scores = value
        self.set_cell(window, class_id, scores=scores)

    @classmethod
    def from_predictions(cls, windows: Iterable['Box'],
                         predictions: Iterable[Any]) -> 'Labels':
        """Overrid to convert predictions to (class_id, scores) pairs."""
        predictions = ((np.argmax(p), p) for p in predictions)
        return super().from_predictions(windows, predictions)

    @classmethod
    def make_empty(cls) -> 'ChipClassificationLabels':
        return ChipClassificationLabels()

    def filter_by_aoi(self, aoi_polygons: Iterable['Polygon']):
        result = ChipClassificationLabels()
        for cell in self.cell_to_label:
            cell_box = Box(*cell)
            cell_poly = cell_box.to_shapely()
            for aoi in aoi_polygons:
                if cell_poly.within(aoi):
                    (class_id, scores) = self.cell_to_label[cell]
                    result.set_cell(cell_box, class_id, scores)
        return result

    def set_cell(self,
                 cell: Box,
                 class_id: int,
                 scores: Optional['np.ndarray'] = None) -> None:
        """Set cell and its class_id.

        Args:
            cell: (Box)
            class_id: int
            scores: 1d numpy array of probabilities for each class
        """
        if scores is not None:
            scores = list(map(lambda x: float(x), list(scores)))
        class_id = int(class_id)
        self.cell_to_label[cell] = ClassificationLabel(class_id, scores)

    def get_cell_class_id(self, cell: Box) -> int:
        """Return class_id for a cell.

        Args:
            cell: (Box)
        """
        result = self.cell_to_label.get(cell)
        if result is not None:
            return result.class_id
        else:
            return None

    def get_cell_scores(self, cell: Box) -> Optional[Sequence[float]]:
        """Return scores for a cell.

        Args:
            cell: (Box)
        """
        result = self.cell_to_label.get(cell)
        if result is not None:
            return result.score
        else:
            return None

    def get_singleton_labels(self, cell: Box):
        """Return Labels object representing a single cell.

        Args:
            cell: (Box)
        """
        return ChipClassificationLabels({cell: self[cell]})

    def get_cells(self) -> List[Box]:
        """Return list of all cells (list of Box)."""
        return list(self.cell_to_label.keys())

    def get_class_ids(self) -> List[int]:
        """Return list of class_ids for all cells."""
        return [label.class_id for label in self.cell_to_label.values()]

    def get_scores(self) -> List[Optional[Sequence[float]]]:
        """Return list of scores for all cells."""
        return [label.scores for label in self.cell_to_label.values()]

    def get_values(self) -> List[ClassificationLabel]:
        """Return list of class_ids and scores for all cells."""
        return list(self.cell_to_label.values())

    def extend(self, labels: 'ChipClassificationLabels') -> None:
        """Adds cells contained in labels.

        Args:
            labels: ChipClassificationLabels
        """
        for cell in labels.get_cells():
            self.set_cell(cell, *labels[cell])

    def save(self, uri: str, class_config: 'ClassConfig',
             crs_transformer: 'CRSTransformer') -> None:
        """Save labels as a GeoJSON file.

        Args:
            uri (str): URI of the output file.
            class_config (ClassConfig): ClassConfig to map class IDs to names.
            crs_transformer (CRSTransformer): CRSTransformer to convert from
                pixel-coords to map-coords before saving.
        """
        from rastervision.core.data import ChipClassificationGeoJSONStore

        label_store = ChipClassificationGeoJSONStore(
            uri=uri,
            class_config=class_config,
            crs_transformer=crs_transformer)
        label_store.save(self)
