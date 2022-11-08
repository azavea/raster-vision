from typing import TYPE_CHECKING
import logging

import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

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
        null_class_id = self.class_config.null_class_id
        num_classes = len(self.class_config)
        labels = np.arange(num_classes)
        self.conf_mat = np.zeros((num_classes, num_classes))
        windows = pred_labels.get_windows()
        with tqdm(windows, delay=5, desc='Computing metrics') as bar:
            for window in bar:
                gt_arr = gt_labels.get_label_arr(window, null_class_id)
                pred_arr = pred_labels.get_label_arr(window, null_class_id)
                self.conf_mat += confusion_matrix(
                    gt_arr.ravel(), pred_arr.ravel(), labels=labels)

        for class_id, class_name in enumerate(self.class_config.names):
            eval_item = ClassEvaluationItem.from_multiclass_conf_mat(
                conf_mat=self.conf_mat,
                class_id=class_id,
                class_name=class_name)
            self.class_to_eval_item[class_id] = eval_item

        self.compute_avg()
