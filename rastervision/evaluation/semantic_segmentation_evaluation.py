import math
import logging
import json

from rastervision.evaluation import ClassEvaluationItem
from rastervision.evaluation import ClassificationEvaluation

log = logging.getLogger(__name__)


def is_geojson(data):
    if isinstance(data, dict):
        return True
    else:
        try:
            json.loads(data)
            retval = True
        except ValueError:
            retval = False
        return retval


class SemanticSegmentationEvaluation(ClassificationEvaluation):
    """Evaluation for semantic segmentation.
    """

    def __init__(self, class_map):
        super().__init__()
        self.class_map = class_map

    def compute_vector(self, gt, pred, mode, class_id):
        import mask_to_polygons.vectorification as vectorification
        import mask_to_polygons.processing.score as score

        # Ground truth as list of geometries
        if is_geojson(gt):
            _ground_truth = gt
            if 'features' in _ground_truth.keys():
                _ground_truth = _ground_truth['features']
            ground_truth = []
            for feature in _ground_truth:
                if 'geometry' in feature.keys():
                    ground_truth.append(feature['geometry'])
                else:
                    ground_truth.append(feature)
        else:
            ground_truth = vectorification.geometries_from_geojson(gt)

        # Predictions as list of geometries
        if is_geojson(pred):
            predictions = pred
        else:
            predictions = vectorification.geometries_from_geojson(pred)

        if len(ground_truth) > 0 and len(predictions) > 0:
            results = score.spacenet(predictions, ground_truth)

            true_positives = results['tp']
            false_positives = results['fp']
            false_negatives = results['fn']
            precision = float(true_positives) / (
                true_positives + false_positives)
            recall = float(true_positives) / (true_positives + false_negatives)
            if precision + recall != 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            count_error = int(false_positives + false_negatives)
            gt_count = len(ground_truth)
            class_name = 'vector-{}'.format(mode)

            evaluation_item = ClassEvaluationItem(precision, recall, f1,
                                                  count_error, gt_count,
                                                  -class_id, class_name)

            if hasattr(self, 'class_to_eval_item') and isinstance(
                    self.class_to_eval_item, dict):
                self.class_to_eval_item[-class_id] = evaluation_item
            else:
                self.class_to_eval_item = {-class_id: evaluation_item}
            self.compute_avg()

    def compute(self, ground_truth_labels, prediction_labels):
        # Definitions of precision, recall, and f1 taken from
        # http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html  # noqa
        ground_truth_labels = ground_truth_labels.to_array()
        prediction_labels = prediction_labels.to_array()

        log.debug('Type of ground truth labels: {}'.format(
            ground_truth_labels.dtype))
        log.debug('Type of prediction labels: {}'.format(
            prediction_labels.dtype))

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
