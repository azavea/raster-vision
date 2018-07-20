from abc import ABC, abstractmethod


class LabelStore(ABC):
    """The place where labels are stored for a scene.

    An LabelStore is a read/write source of labels for a scene
    that could be backed by a file, a database, an API, etc. The difference
    between LabelStores and Labels can be understood by analogy to the
    difference between a database and result sets queried from a database.
    """

    @abstractmethod
    def clear(self):
        """Clear all labels."""
        pass

    @abstractmethod
    def get_labels(self, window=None):
        """Return labels overlapping with window.

        Args:
            window: Box

        Returns:
            Labels overlapping with window. If window is None,
                returns all labels.
        """
        pass

    @abstractmethod
    def set_labels(self, labels):
        """Set labels for the LabelStore.

        Args:
            labels: Labels
        """
        pass

    @abstractmethod
    def extend(self, labels):
        """Add labels to the LabelStore.

        Args:
            labels: Labels
        """
        pass

    @abstractmethod
    def save(self):
        """Save."""
        pass
