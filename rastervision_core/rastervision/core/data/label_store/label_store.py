from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from rastervision.core.box import Box
    from rastervision.core.data import CRSTransformer


class LabelStore(ABC):
    """This defines how to store prediction labels for a scene."""

    @abstractmethod
    def save(self, labels):
        """Save.

        Args:
           labels - Labels to be saved, the type of which will be dependent on the type
                    of pipeline.
        """

    @abstractmethod
    def get_labels(self):
        """Loads Labels from this label store."""

    @property
    @abstractmethod
    def bbox(self) -> 'Box | None':
        """Bounding box applied to the source."""

    @property
    @abstractmethod
    def crs_transformer(self) -> 'CRSTransformer':
        """Associated :class:`.CRSTransformer`."""

    @abstractmethod
    def set_bbox(self, extent: 'Box') -> None:
        """Set self.extent to the given value.

        .. note:: This method is idempotent.

        Args:
            extent (Box): User-specified extent in pixel coordinates.
        """
