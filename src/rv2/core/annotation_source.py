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
        pass

    @abstractmethod
    def get_all_annotations(self):
        pass

    @abstractmethod
    def extend(self, window, annotations):
        pass

    @abstractmethod
    def post_process(self):
        pass

    @abstractmethod
    def save(self, label_map):
        pass

    @abstractmethod
    def clear(self):
        pass
