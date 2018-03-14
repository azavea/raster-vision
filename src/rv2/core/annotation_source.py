from abc import ABC, abstractmethod


class AnnotationSource(ABC):
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
