from abc import ABC, abstractmethod


class Evaluation(ABC):
    """A way of evaluating the predictions for a set of projects."""
    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def compute(ground_truth_annotation_source, prediction_annotation_source):
        pass

    @abstractmethod
    def merge(self, evaluation):
        pass

    @abstractmethod
    def save(self, output_uri):
        pass
