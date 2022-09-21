from typing import TYPE_CHECKING
import logging

from sklearn.metrics import confusion_matrix
import numpy as np

from rastervision.core.evaluation import ClassEvaluationItem
from rastervision.core.evaluation import ClassificationEvaluation

if TYPE_CHECKING:
    from rastervision.core.data import (ClassConfig,
                                        SemanticSegmentationLabels)

log = logging.getLogger(__name__)


class SemanticSegmentationEvaluation(ClassificationEvaluation):
    """Evaluation for semantic segmentation."""

    def __init__(self, class_config: 'ClassConfig'):
        super().__init__()
        self.class_config = class_config

    def compute(self, gt_labels: 'SemanticSegmentationLabels',
                pred_labels: 'SemanticSegmentationLabels') -> None:
        self.reset()

        # compute confusion matrix
        num_classes = len(self.class_config)
        labels = np.arange(num_classes)
        self.conf_mat = np.zeros((num_classes, num_classes))
        for window in pred_labels.get_windows():
            log.debug(f'Evaluating window: {window}')
            gt_arr = gt_labels.get_label_arr(window).ravel()
            pred_arr = pred_labels.get_label_arr(window).ravel()
            self.conf_mat += confusion_matrix(gt_arr, pred_arr, labels=labels)

        for class_id, class_name in enumerate(self.class_config.names):
            eval_item = ClassEvaluationItem.from_multiclass_conf_mat(
                conf_mat=self.conf_mat,
                class_id=class_id,
                class_name=class_name)
            self.class_to_eval_item[class_id] = eval_item

        self.compute_avg()
