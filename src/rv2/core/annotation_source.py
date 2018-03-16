from abc import ABC, abstractmethod


class AnnotationSource(ABC):
    """A source of annotations for a project.

    An AnnotationSource is a read/write source of annotations for a project
    that could be backed by a file, a database, an API, etc.
    """

    @abstractmethod
    def get_annotations(self, window):
        """Get annotations for a window.

        Args:
            window: Box covering area to retrieve Annotations from

        Returns:
            Annotations object with annotations lying inside the window
        """
        pass

    @abstractmethod
    def get_all_annotations(self):
        """Get all annotations.

        Returns:
            Annotations object with all the annotations.
        """
        pass

    @abstractmethod
    def extend(self, window, annotations):
        """Add annotations to the AnnotationSource.

        Args:
            window: Box covering area where annotations are from
            annotations: Annotations
        """
        pass

    @abstractmethod
    def post_process(self):
        """Perform some preprocessing operation before saving."""
        pass

    @abstractmethod
    def save(self, label_map):
        """Save.

        Args:
            label_map: LabelMap
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear all annotations."""
        pass
