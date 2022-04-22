from typing import TYPE_CHECKING, Any, Dict, Optional
from abc import (ABC, abstractmethod)
import copy
import json

import numpy as np

from rastervision.pipeline.file_system import str_to_file

if TYPE_CHECKING:
    from rastervision.core.evaluation import ClassEvaluationItem


class ClassificationEvaluation(ABC):
    """Base class for evaluating predictions for pipelines that have classes.

    Evaluations can be keyed, for instance, if evaluations happen per class.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the Evaluation."""
        self.class_to_eval_item: Dict[int, 'ClassEvaluationItem'] = {}
        self.scene_to_eval: Dict[str, 'ClassificationEvaluation'] = {}
        self.avg_item: Optional[Dict[str, Any]] = None
        self.conf_mat: Optional[np.ndarray] = None
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
            json_rep.append(self.avg_item)

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
        json_str = json.dumps(
            ensure_json_serializable(self.to_json()), indent=4)
        str_to_file(json_str, output_uri)

    def merge(self, other: 'ClassificationEvaluation', scene_id=None) -> None:
        """Merge Evaluation for another Scene into this one.

        This is useful for computing the average metrics of a set of scenes.
        The results of the averaging are stored in this Evaluation.

        Args:
            other: Evaluation to merge into this one
        """
        if self.conf_mat is None:
            self.conf_mat = other.conf_mat
        else:
            self.conf_mat += other.conf_mat

        if len(self.class_to_eval_item) == 0:
            self.class_to_eval_item = other.class_to_eval_item
        else:
            for key, other_eval_item in other.class_to_eval_item.items():
                if self.has_id(key):
                    self.get_by_id(key).merge(other_eval_item)
                else:
                    self.class_to_eval_item[key] = other_eval_item

        self._is_empty = False
        self.compute_avg()

        if scene_id is not None:
            self.scene_to_eval[scene_id] = copy.deepcopy(other)

    def compute_avg(self):
        """Compute average metrics over all keys."""
        if len(self.class_to_eval_item) == 0:
            return
        class_evals = [
            eval_item.to_json()
            for eval_item in self.class_to_eval_item.values()
        ]
        # compute weighted averages of metrics
        class_counts = np.array([e['gt_count'] for e in class_evals])
        class_weights = class_counts / class_counts.sum()
        class_metrics = [e['metrics'] for e in class_evals]
        metric_names = class_metrics[0].keys()
        avg_metrics = {}
        for k in metric_names:
            metric_vals = np.array([m[k] for m in class_metrics], dtype=float)
            avg_metrics[k] = np.nan_to_num(metric_vals * class_weights).sum()

        # sum the counts
        gt_count = sum(e['gt_count'] for e in class_evals)
        pred_count = sum(e['pred_count'] for e in class_evals)
        count_error = sum(e['count_error'] for e in class_evals)

        self.avg_item = {
            'class_name': 'average',
            'metrics': avg_metrics,
            'gt_count': gt_count,
            'pred_count': pred_count,
            'count_error': count_error
        }
        if self.conf_mat is not None:
            self.avg_item['conf_mat'] = self.conf_mat.tolist()

    @abstractmethod
    def compute(self, ground_truth_labels, prediction_labels):
        """Compute metrics for a single scene.

        Args:
            ground_truth_labels: Ground Truth labels to evaluate against.
            prediction_labels: The predicted labels to evaluate.
        """
        pass


def ensure_json_serializable(obj: Any) -> dict:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(o) for o in obj]
    if isinstance(obj, np.ndarray):
        return ensure_json_serializable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj
