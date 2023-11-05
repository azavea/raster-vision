"""Defines abstract base evaluation class for all tasks."""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from abc import (ABC, abstractmethod)
import copy
import json

import numpy as np

from rastervision.pipeline.file_system import str_to_file
from rastervision.core.data.utils import ensure_json_serializable

if TYPE_CHECKING:
    from rastervision.core.evaluation import ClassEvaluationItem


class ClassificationEvaluation(ABC):
    """Base class for representing prediction evaluations.

    Evaluations can be keyed, for instance, if evaluations happen per class.

    Attributes:
        class_to_eval_item (Dict[int, ClassEvaluationItem]): Mapping from class
            IDs to ``ClassEvaluationItem``s.
        scene_to_eval (Dict[str, ClassificationEvaluation]): Mapping from scene
            IDs to ``ClassificationEvaluation``s.
        avg_item (Optional[Dict[str, Any]]): Averaged evaluation over all
            classes.
        conf_mat (Optional[np.ndarray]): Confusion matrix.
    """

    def __init__(self):
        self.class_to_eval_item: Dict[int, 'ClassEvaluationItem']
        self.scene_to_eval: Dict[str, 'ClassificationEvaluation']
        self.avg_item: Optional[Dict[str, Any]]
        self.conf_mat: Optional[np.ndarray]
        self.reset()

    def reset(self):
        """Reset the Evaluation."""
        self.class_to_eval_item = {}
        self.scene_to_eval = {}
        self.avg_item = None
        self.conf_mat = None

    def to_json(self) -> Union[dict, list]:
        """Serialize to a dict or list.

        Returns:
            Union[dict, list]: Class-wise and (if available) scene-wise
            evaluations.
        """
        out = [item.to_json() for item in self.class_to_eval_item.values()]
        if self.avg_item:
            out.append(self.avg_item)

        if len(self.scene_to_eval) > 0:
            # append per scene evals
            out = {'overall': out}
            per_scene_evals = {
                scene_id: eval.to_json()
                for scene_id, eval in self.scene_to_eval.items()
            }
            out['per_scene'] = per_scene_evals

        return out

    def save(self, output_uri: str) -> None:
        """Save this Evaluation to a file.

        Args:
            output_uri: string URI for the file to write.
        """
        json_str = json.dumps(
            ensure_json_serializable(self.to_json()), indent=4)
        str_to_file(json_str, output_uri)

    def merge(self,
              other: 'ClassificationEvaluation',
              scene_id: Optional[str] = None) -> None:
        """Merge Evaluation for another Scene into this one.

        This is useful for computing the average metrics of a set of scenes.
        The results of the averaging are stored in this Evaluation.

        Args:
            other (ClassificationEvaluation): Evaluation to merge into this one
            scene_id (Optional[str], optional): ID of scene. If specified,
                (a copy of) ``other`` will be saved and be available in
                ``to_json()``'s output. Defaults to None.
        """
        if self.conf_mat is None:
            self.conf_mat = other.conf_mat
        else:
            self.conf_mat += other.conf_mat

        if len(self.class_to_eval_item) == 0:
            self.class_to_eval_item = other.class_to_eval_item
        else:
            for class_id, other_eval_item in other.class_to_eval_item.items():
                if class_id in self.class_to_eval_item:
                    self.class_to_eval_item[class_id].merge(other_eval_item)
                else:
                    self.class_to_eval_item[class_id] = other_eval_item

        self.compute_avg()

        if scene_id is not None:
            self.scene_to_eval[scene_id] = copy.deepcopy(other)

    def compute_avg(self) -> None:
        """Compute average metrics over all classes."""
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
            cm = self.conf_mat
            self.avg_item['conf_mat'] = cm.tolist()
            self.avg_item['conf_mat_frac'] = (cm / cm.sum()).tolist()

    @abstractmethod
    def compute(self, ground_truth_labels, prediction_labels):
        """Compute metrics for a single scene.

        Args:
            ground_truth_labels: Ground Truth labels to evaluate against.
            prediction_labels: The predicted labels to evaluate.
        """
