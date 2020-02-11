import numpy as np
from sklearn import metrics

from rastervision2.core.evaluation import (ClassificationEvaluation,
                                           ClassEvaluationItem)


class ChipClassificationEvaluation(ClassificationEvaluation):
    def __init__(self, class_config):
        super().__init__()
        self.class_config = class_config

    def compute(self, ground_truth_labels, prediction_labels):
        self.class_to_eval_item = ChipClassificationEvaluation.compute_eval_items(
            ground_truth_labels, prediction_labels, self.class_config)

        self.compute_avg()

    @staticmethod
    def compute_eval_items(gt_labels, pred_labels, class_config):
        nb_classes = len(class_config.names)
        class_to_eval_item = {}

        gt_class_ids = []
        pred_class_ids = []

        gt_cells = gt_labels.get_cells()
        for gt_cell in gt_cells:
            gt_class_id = gt_labels.get_cell_class_id(gt_cell)
            pred_class_id = pred_labels.get_cell_class_id(gt_cell)

            if gt_class_id is not None and pred_class_id is not None:
                gt_class_ids.append(gt_class_id)
                pred_class_ids.append(pred_class_id)

        sklabels = np.arange(nb_classes)
        precision, recall, f1, support = metrics.precision_recall_fscore_support(
            gt_class_ids, pred_class_ids, labels=sklabels, warn_for=())

        for class_id, class_name in enumerate(class_config.names):
            eval_item = ClassEvaluationItem(
                float(precision[class_id]),
                float(recall[class_id]),
                float(f1[class_id]),
                gt_count=float(support[class_id]),
                class_id=class_id,
                class_name=class_name)
            class_to_eval_item[class_id] = eval_item

        return class_to_eval_item
