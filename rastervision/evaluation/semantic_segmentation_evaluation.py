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


def get_class_eval_item(gt_arr, pred_arr, class_id, class_map):
    class_name = class_map.get_by_id(class_id).name

    if gt_arr.sum() == 0:
        return ClassEvaluationItem(None, None, None, 0, 0, class_id,
                                   class_name)

    # Definitions of precision, recall, and f1 taken from
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html  # noqa
    not_dont_care = (gt_arr != 0)  # By assumption
    gt = (gt_arr == class_id)
    pred = (pred_arr == class_id)
    not_gt = (gt_arr != class_id)
    not_pred = (pred_arr != class_id)

    true_pos = (gt * pred).sum()
    false_pos = (not_gt * pred * not_dont_care).sum()
    false_neg = (gt * not_pred * not_dont_care).sum()

    precision = float(true_pos) / (true_pos + false_pos)
    recall = float(true_pos) / (true_pos + false_neg)
    f1 = 2 * (precision * recall) / (precision + recall)
    count_error = int(false_pos + false_neg)
    gt_count = int(gt.sum())

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

    return ClassEvaluationItem(precision, recall, f1, count_error, gt_count,
                               class_id, class_name)


class SemanticSegmentationEvaluation(ClassificationEvaluation):
    """Evaluation for semantic segmentation."""

    def __init__(self, class_map):
        super().__init__()
        self.class_map = class_map

    def compute(self, gt_labels, pred_labels):
        self.clear()
        for window in pred_labels.get_windows():
            log.debug('Evaluating window: {}'.format(window))
            gt_arr = gt_labels.get_label_arr(window)
            pred_arr = pred_labels.get_label_arr(window)

            eval_items = []
            for class_id in self.class_map.get_keys():
                eval_item = get_class_eval_item(gt_arr, pred_arr, class_id,
                                                self.class_map)
                eval_items.append(eval_item)

            # Treat each window as if it was a small Scene.
            window_eval = SemanticSegmentationEvaluation(self.class_map)
            window_eval.set_class_to_eval_item(
                dict(zip(self.class_map.get_keys(), eval_items)))
            window_eval.compute_avg()
            self.merge(window_eval)

    def compute_vector(self, gt, pred, mode, class_id):
        """Compute evaluation over vector predictions.
            Args:
                gt: Ground-truth GeoJSON.  Either a string (containing
                    unparsed GeoJSON or a file name), or a dictionary
                    containing parsed GeoJSON.
                pred: GeoJSON for predictions.  Either a string
                    (containing unparsed GeoJSON or a file name), or a
                    dictionary containing parsed GeoJSON.
                mode: A string containing either 'buildings' or
                    'polygons'.
                class_id: An integer containing the class id of
                    interest.
        """
        import mask_to_polygons.vectorification as vectorification
        import mask_to_polygons.processing.score as score

        # Ground truth as list of geometries
        def get_geoms(x):
            if is_geojson(x):
                _x = x
                if 'features' in _x.keys():
                    _x = _x['features']
                geoms = []
                for feature in _x:
                    if 'geometry' in feature.keys():
                        geoms.append(feature['geometry'])
                    else:
                        geoms.append(feature)
            else:
                geoms = vectorification.geometries_from_geojson(x)

            return geoms

        gt = get_geoms(gt)
        pred = get_geoms(pred)

        if len(gt) > 0 and len(pred) > 0:
            results = score.spacenet(pred, gt)

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
            gt_count = len(gt)
            class_name = 'vector-{}-{}'.format(
                mode,
                self.class_map.get_by_id(class_id).name)

            evaluation_item = ClassEvaluationItem(precision, recall, f1,
                                                  count_error, gt_count,
                                                  class_id, class_name)

            if hasattr(self, 'class_to_eval_item') and isinstance(
                    self.class_to_eval_item, dict):
                self.class_to_eval_item[class_id] = evaluation_item
            else:
                self.class_to_eval_item = {class_id: evaluation_item}
            self.compute_avg()
