from abc import ABC, abstractmethod


class Evaluation(ABC):
    """An evaluation of the predictions for a set of projects."""

    @abstractmethod
    def clear(self):
        """Clear the Evaluation."""
        pass

    @abstractmethod
    def compute(ground_truth_label_source, prediction_label_source):
        """Compute metrics for a single project.

        Args:
            ground_truth_label_source: LabelSource with the ground
                truth
            prediction_label_source: LabelSource with the
                corresponding predictions
        """
        pass

    @abstractmethod
    def merge(self, evaluation):
        """Merge Evaluation for another Project into this one.

        This is useful for computing the average metrics of a set of projects.
        The results of the averaging are stored in this Evaluation.

        Args:
            evaluation: Evaluation to merge into this one
        """
        pass

    @abstractmethod
    def save(self, output_uri):
        """Save this Evaluation to a file.

        Args:
            output_uri: string URI for the file to write.
        """
        pass
