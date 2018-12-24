from abc import ABC, abstractmethod

from rastervision.data import ActivateMixin


class LabelStore(ABC, ActivateMixin):
    """This defines how to store prediction labels are stored for a scene.
    """

    @abstractmethod
    def save(self, labels):
        """Save.

        Args:
           labels - Labels to be saved, the type of which will be dependant on the type
                    of task.
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

    def _subcomponents_to_activate(self):
        pass

    def _activate(self):
        pass

    def _deactivate(self):
        pass
