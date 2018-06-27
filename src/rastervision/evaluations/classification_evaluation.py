import numpy as np
from sklearn import metrics

from rastervision.core.evaluation import Evaluation
from rastervision.core.evaluation_item import EvaluationItem


def compute_eval_items(gt_labels, pred_labels, class_map):
    nb_classes = len(class_map)
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

    # Add 1 because class_ids start at 1.
    sklabels = np.arange(1 + nb_classes)
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        gt_class_ids, pred_class_ids, labels=sklabels, warn_for=())

    for class_map_item in class_map.get_items():
        class_id = class_map_item.id
        class_name = class_map_item.name

        eval_item = EvaluationItem(
            float(precision[class_id]), float(recall[class_id]),
            float(f1[class_id]), gt_count=float(support[class_id]),
            class_id=class_id, class_name=class_name)
        class_to_eval_item[class_id] = eval_item

    return class_to_eval_item


class ClassificationEvaluation(Evaluation):
    def compute(self, class_map, ground_truth_label_store,
                prediction_label_store):
        gt_labels = ground_truth_label_store.get_all_labels()
        pred_labels = prediction_label_store.get_all_labels()

        self.class_to_eval_item = compute_eval_items(
            gt_labels, pred_labels, class_map)

        self.compute_avg()

    def compute_avg(self):
        self.avg_item = EvaluationItem(class_name='average')
        for eval_item in self.class_to_eval_item.values():
            self.avg_item.merge(eval_item)
