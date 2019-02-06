from abc import (ABC, abstractmethod)
import copy

import json

from rastervision.evaluation import ClassEvaluationItem
from rastervision.utils.files import str_to_file


class ClassificationEvaluation(ABC):
    """Base class for evaluating predictions for tasks that have classes.

    Evaluations can be keyed, for instance, if evaluations happen per class.
    """

    def __init__(self):
        self.clear()
        self._is_empty = True

    def clear(self):
        """Clear the Evaluation."""
        self.class_to_eval_item = {}
        self.scene_to_eval = {}
        self.avg_item = None
        self._is_empty = True

    def is_empty(self):
        return self._is_empty

    def set_class_to_eval_item(self, class_to_eval_item):
        self.class_to_eval_item = class_to_eval_item

    def get_by_id(self, key):
        """Gets the evaluation for a particular EvaluationItem key"""
        return self.class_to_eval_item[key]

    def has_id(self, key):
        """Answers whether or not the EvaluationItem key is represented"""
        return key in self.class_to_eval_item

    def to_json(self):
        json_rep = []
        for eval_item in self.class_to_eval_item.values():
            json_rep.append(eval_item.to_json())
        if self.avg_item:
            json_rep.append(self.avg_item.to_json())

        if self.scene_to_eval:
            json_rep = {'overall': json_rep}
            scene_to_eval_json = {}
            for scene_id, eval in self.scene_to_eval.items():
                scene_to_eval_json[scene_id] = eval.to_json()
            json_rep['per_scene'] = scene_to_eval_json

        return json_rep

    def save(self, output_uri):
        """Save this Evaluation to a file.

        Args:
            output_uri: string URI for the file to write.
        """
        json_str = json.dumps(self.to_json(), indent=4)
        str_to_file(json_str, output_uri)

    def merge(self, evaluation, scene_id=None):
        """Merge Evaluation for another Scene into this one.

        This is useful for computing the average metrics of a set of scenes.
        The results of the averaging are stored in this Evaluation.

        Args:
            evaluation: Evaluation to merge into this one
        """
        if len(self.class_to_eval_item) == 0:
            self.class_to_eval_item = evaluation.class_to_eval_item
        else:
            for key, other_eval_item in \
                    evaluation.class_to_eval_item.items():
                if self.has_id(key):
                    self.get_by_id(key).merge(other_eval_item)
                else:
                    self.class_to_eval_item[key] = other_eval_item

        self._is_empty = False
        self.compute_avg()

        if scene_id is not None:
            self.scene_to_eval[scene_id] = copy.deepcopy(evaluation)

    def compute_avg(self):
        """Compute average metrics over all keys."""
        self.avg_item = ClassEvaluationItem(class_name='average')
        for eval_item in self.class_to_eval_item.values():
            self.avg_item.merge(eval_item)

    @abstractmethod
    def compute(self, ground_truth_labels, prediction_labels):
        """Compute metrics for a single scene.

        Args:
            ground_truth_labels: Ground Truth labels to evaluate against.
            prediction_labels: The predicted labels to evaluate.
        """
        pass
