from abc import ABC, abstractmethod


class LabelSource(ABC):
    """An interface for storage of labels for a scene.

    An LabelSource is a read source of labels for a scene
    that could be backed by a file, a database, an API, etc. The difference
    between LabelSources and Labels can be understood by analogy to the
    difference between a database and result sets queried from a database.
    """

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
