from abc import ABC, abstractmethod


class LabelStore(ABC):
    """This defines how to store prediction labels for a scene."""

    @abstractmethod
    def save(self, labels):
        """Save.

        Args:
           labels - Labels to be saved, the type of which will be dependant on the type
                    of pipeline.
        """
        pass

    @abstractmethod
    def get_labels(self):
        """Loads Labels from this label store."""
        pass

    @abstractmethod
    def empty_labels(self):
        """Produces an empty Labels"""
        pass
