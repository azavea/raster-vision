from abc import ABC, abstractmethod
import json

from rastervision.utils.files import str_to_file


class Evaluation(ABC):
    """An evaluation of the predictions for a set of scenes."""

    def __init__(self):
        self.clear()

    def clear(self):
        """Clear the Evaluation."""
        self.class_to_eval_item = {}
        self.avg_item = None

    def get_by_id(self, class_id):
        return self.class_to_eval_item[class_id]

    def to_json(self):
        json_rep = []
        for eval_item in self.class_to_eval_item.values():
            json_rep.append(eval_item.to_json())
        json_rep.append(self.avg_item.to_json())
        return json_rep

    def save(self, output_uri):
        """Save this Evaluation to a file.

        Args:
            output_uri: string URI for the file to write.
        """
        json_str = json.dumps(self.to_json(), indent=4)
        str_to_file(json_str, output_uri)

    def merge(self, evaluation):
        """Merge Evaluation for another Scene into this one.

        This is useful for computing the average metrics of a set of scenes.
        The results of the averaging are stored in this Evaluation.

        Args:
            evaluation: Evaluation to merge into this one
        """
        if len(self.class_to_eval_item) == 0:
            self.class_to_eval_item = evaluation.class_to_eval_item
        else:
            for class_id, other_eval_item in evaluation.class_to_eval_item.items(
            ):
                self.get_by_id(class_id).merge(other_eval_item)

        self.compute_avg()

    @abstractmethod
    def compute(self, ground_truth_label_store, prediction_label_store):
        """Compute metrics for a single scene.

        Args:
            ground_truth_label_store: LabelStore with the ground
                truth
            prediction_label_store: LabelStore with the
                corresponding predictions
        """
        pass

    @abstractmethod
    def compute_avg(self):
        """Compute average metrics over classes."""
        pass
