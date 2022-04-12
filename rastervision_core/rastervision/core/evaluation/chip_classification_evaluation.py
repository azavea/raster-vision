from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import confusion_matrix

from rastervision.core.evaluation import (ClassificationEvaluation,
                                          ClassEvaluationItem)
if TYPE_CHECKING:
    from rastervision.core.data import (ChipClassificationLabels, ClassConfig)


class ChipClassificationEvaluation(ClassificationEvaluation):
    def __init__(self, class_config: 'ClassConfig'):
        super().__init__()
        self.class_config = class_config

    def compute(self, gt_labels: 'ChipClassificationLabels',
                pred_labels: 'ChipClassificationLabels') -> None:
        self.reset()
        self.class_to_eval_item = {}

        gt_class_ids = []
        pred_class_ids = []
        for gt_cell in gt_labels.get_cells():
            gt_class_id = gt_labels.get_cell_class_id(gt_cell)
            pred_class_id = pred_labels.get_cell_class_id(gt_cell)

            if gt_class_id is not None and pred_class_id is not None:
                gt_class_ids.append(gt_class_id)
                pred_class_ids.append(pred_class_id)

        labels = np.arange(len(self.class_config))
        self.conf_mat = confusion_matrix(
            gt_class_ids, pred_class_ids, labels=labels)

        for class_id, class_name in enumerate(self.class_config.names):
            eval_item = ClassEvaluationItem.from_multiclass_conf_mat(
                conf_mat=self.conf_mat,
                class_id=class_id,
                class_name=class_name)
            self.class_to_eval_item[class_id] = eval_item

        self.compute_avg()
