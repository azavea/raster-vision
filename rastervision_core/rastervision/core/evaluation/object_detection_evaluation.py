import numpy as np
import shapely
import shapely.strtree
import shapely.geometry

from rastervision.core.data import ObjectDetectionLabels
from rastervision.core.evaluation import ClassEvaluationItem
from rastervision.core.evaluation import ClassificationEvaluation


def compute_metrics(gt_labels: ObjectDetectionLabels,
                    pred_labels: ObjectDetectionLabels,
                    num_classes: int,
                    iou_thresh=0.5):
    gt_geoms = [b.to_shapely() for b in gt_labels.get_boxes()]
    gt_classes = gt_labels.get_class_ids()
    pred_geoms = [b.to_shapely() for b in pred_labels.get_boxes()]
    pred_classes = pred_labels.get_class_ids()

    for pred, class_id in zip(pred_geoms, pred_classes):
        pred.class_id = class_id
    pred_tree = shapely.strtree.STRtree(pred_geoms)

    def iou(a, b):
        return a.intersection(b).area / a.union(b).area

    def is_matched(geom):
        return hasattr(geom, 'iou_matched')

    tp = np.zeros((num_classes, ))
    fp = np.zeros((num_classes, ))
    fn = np.zeros((num_classes, ))

    for gt, gt_class in zip(gt_geoms, gt_classes):
        matches = list(
            filter(lambda g: (not is_matched(g)) and g.class_id == gt_class,
                   pred_tree.query(gt)))
        scores = [iou(m, gt) for m in matches]
        if len(scores) > 0:
            max_ind = np.argmax(scores)
            if scores[max_ind] > iou_thresh:
                matches[max_ind].iou_matched = True
                tp[gt_class] += 1
            else:
                fn[gt_class] += 1
        else:
            fn[gt_class] += 1

    for class_id in range(num_classes):
        pred_not_matched = np.array([not is_matched(g) for g in pred_geoms])
        fp[class_id] = np.sum(pred_not_matched[pred_classes == class_id])

    return tp, fp, fn


class ObjectDetectionEvaluation(ClassificationEvaluation):
    def __init__(self, class_config):
        super().__init__()
        self.class_config = class_config

    def compute(self, ground_truth_labels, prediction_labels):
        self.class_to_eval_item = ObjectDetectionEvaluation.compute_eval_items(
            ground_truth_labels, prediction_labels, self.class_config)
        self.compute_avg()

    @staticmethod
    def compute_eval_items(gt_labels, pred_labels, class_config):
        iou_thresh = 0.5
        num_classes = len(class_config)
        tps, fps, fns = compute_metrics(gt_labels, pred_labels, num_classes,
                                        iou_thresh)
        class_to_eval_item = {}

        for class_id, (tp, fp, fn) in enumerate(zip(tps, fps, fns)):
            gt_count = tp + fn
            pred_count = tp + fp
            class_name = class_config.get_name(class_id)

            if gt_count == 0:
                eval_item = ClassEvaluationItem(
                    class_id=class_id, class_name=class_name)
            elif pred_count == 0:
                eval_item = ClassEvaluationItem(
                    precision=None,
                    recall=0,
                    gt_count=gt_count,
                    class_id=class_id,
                    class_name=class_name)
            else:
                prec = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 0.
                if prec + recall != 0.0:
                    f1 = 2 * (prec * recall) / (prec + recall)
                count_err = pred_count - gt_count
                norm_count_err = None
                if gt_count > 0:
                    norm_count_err = count_err / gt_count

                eval_item = ClassEvaluationItem(
                    precision=prec,
                    recall=recall,
                    f1=f1,
                    count_error=norm_count_err,
                    gt_count=gt_count,
                    class_id=class_id,
                    class_name=class_name)

            class_to_eval_item[class_id] = eval_item

        return class_to_eval_item
