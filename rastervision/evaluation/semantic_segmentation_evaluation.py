import math

from rastervision.evaluation import ClassEvaluationItem
from rastervision.evaluation import ClassificationEvaluation


class SemanticSegmentationEvaluation(ClassificationEvaluation):
    """Evaluation for semantic segmentation.
    """

    def __init__(self, class_map):
        super().__init__()
        self.class_map = class_map

    def compute(self, ground_truth_labels, prediction_labels):
        # Definitions of precision, recall, and f1 taken from
        # http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html  # noqa
        ground_truth_labels = ground_truth_labels.to_array()
        prediction_labels = prediction_labels.to_array()
        # This shouldn't happen, but just in case...
        if ground_truth_labels.shape != prediction_labels.shape:
            raise ValueError(
                'ground_truth_labels and prediction_labels need to '
                'have the same shape.')

        evaluation_items = []
        for class_id in self.class_map.get_keys():
            not_dont_care = (ground_truth_labels != 0)  # By assumption
            gt = (ground_truth_labels == class_id)
            pred = (prediction_labels == class_id)
            not_gt = (ground_truth_labels != class_id)
            not_pred = (prediction_labels != class_id)

            true_positives = (gt * pred).sum()
            false_positives = (not_gt * pred * not_dont_care).sum()
            false_negatives = (gt * not_pred * not_dont_care).sum()

            precision = float(true_positives) / (
                true_positives + false_positives)
            recall = float(true_positives) / (true_positives + false_negatives)
            f1 = 2 * (precision * recall) / (precision + recall)
            count_error = int(false_positives + false_negatives)
            gt_count = int(gt.sum())
            class_name = self.class_map.get_by_id(class_id).name

            if math.isnan(precision):
                precision = None
            else:
                precision = float(precision)
            if math.isnan(recall):
                recall = None
            else:
                recall = float(recall)
            if math.isnan(f1):
                f1 = None
            else:
                f1 = float(f1)

            evaluation_item = ClassEvaluationItem(precision, recall, f1,
                                                  count_error, gt_count,
                                                  class_id, class_name)
            evaluation_items.append(evaluation_item)

        self.class_to_eval_item = dict(
            zip(self.class_map.get_keys(), evaluation_items))
        self.compute_avg()
