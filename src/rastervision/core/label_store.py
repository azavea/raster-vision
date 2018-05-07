from abc import ABC, abstractmethod
import copy


class LabelStore(ABC):
    """A source of labels for a project.

    An LabelStore is a read/write source of labels for a project
    that could be backed by a file, a database, an API, etc.
    """

    @abstractmethod
    def get_labels(self, window):
        """Get labels for a window.

        Args:
            window: Box covering area to retrieve Labels from

        Returns:
            Labels object with labels lying inside the window
        """
        pass

    @abstractmethod
    def get_all_labels(self):
        """Get all labels.

        Returns:
            Labels object with all the labels.
        """
        pass

    @abstractmethod
    def extend(self, window, labels):
        """Add labels to the LabelStore.

        Args:
            window: Box covering area where labels are from
            labels: Labels
        """
        pass

    @abstractmethod
    def post_process(self):
        """Perform some preprocessing operation before saving."""
        pass

    @abstractmethod
    def save(self):
        """Save."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all labels."""
        pass
