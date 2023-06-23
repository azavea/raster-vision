from typing import TYPE_CHECKING, Any, Optional
from abc import ABC, abstractmethod, abstractproperty

from rastervision.core.box import Box
from rastervision.core.data.utils import parse_array_slices_2d

if TYPE_CHECKING:
    from rastervision.core.data import CRSTransformer, Labels


class LabelSource(ABC):
    """An interface for storage of labels for a scene.

    A LabelSource is a read-only source of labels for a scene
    that could be backed by a file, a database, an API, etc. The difference
    between LabelSources and Labels can be understood by analogy to the
    difference between a database and result sets queried from a database.
    """

    @abstractmethod
    def get_labels(self, window: Optional['Box'] = None) -> 'Labels':
        """Return labels overlapping with window.

        Args:
            window: Box

        Returns:
            Labels overlapping with window. If window is None,
                returns all labels.
        """
        pass

    @property
    def extent(self) -> 'Box':
        """Extent of the ``LabelSource``."""
        return self.bbox.extent

    @abstractproperty
    def bbox(self) -> 'Box':
        """Bounding box applied to the source."""
        pass

    @abstractproperty
    def crs_transformer(self) -> 'CRSTransformer':
        pass

    @abstractmethod
    def set_bbox(self, extent: 'Box') -> None:
        """Set self.extent to the given value.

        .. note:: This method is idempotent.

        Args:
            extent (Box): User-specified extent in pixel coordinates.
        """
        pass

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, Box):
            raise NotImplementedError()
        window, _ = parse_array_slices_2d(key, extent=self.extent)
        return self[window]
