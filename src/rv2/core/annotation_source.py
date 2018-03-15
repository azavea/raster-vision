from abc import ABC, abstractmethod


class AnnotationSource(ABC):
    """A source of annotations for a project.

    An AnnotationSource is a read/write source of annotations for a project
    that could be backed by a file, a database, an API, etc. An example of an
    annotation is a bounding box in the case of object detection or a raster in
    case of segmentation. It is something that is predicted by a model or
    provided by human annotators for the sake of training.
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

    # TODO just absorb this into save?
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
