from typing import TYPE_CHECKING, Dict, Tuple
import numpy as np
from shapely.strtree import STRtree

from rastervision.core.evaluation import (ClassificationEvaluation,
                                          ClassEvaluationItem)

if TYPE_CHECKING:
    from shapely.geometry import Polygon
    from rastervision.core.data import ObjectDetectionLabels
    from rastervision.core.data.class_config import ClassConfig


def compute_metrics(
        gt_labels: 'ObjectDetectionLabels',
        pred_labels: 'ObjectDetectionLabels',
        num_classes: int,
        iou_thresh: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_geoms = [b.to_shapely() for b in gt_labels.get_boxes()]
    gt_classes = gt_labels.get_class_ids()
    pred_geoms = [b.to_shapely() for b in pred_labels.get_boxes()]
    pred_classes = pred_labels.get_class_ids()

    for pred_geom, class_id in zip(pred_geoms, pred_classes):
        pred_geom.class_id = class_id
    pred_tree = STRtree(pred_geoms)

    def iou(a: 'Polygon', b: 'Polygon') -> float:
        return a.intersection(b).area / a.union(b).area

    def is_matched(geom) -> bool:
        return hasattr(geom, 'iou_matched')

    tp = np.zeros((num_classes, ))
    fp = np.zeros((num_classes, ))
    fn = np.zeros((num_classes, ))

    for gt_geom, gt_class in zip(gt_geoms, gt_classes):
        matches = [
            g for g in pred_tree.query(gt_geom)
            if (not is_matched(g)) and (g.class_id == gt_class)
        ]
        ious = np.array([iou(m, gt_geom) for m in matches])
        if (ious > iou_thresh).any():
            max_ind = np.argmax(ious)
            matches[max_ind].iou_matched = True
            tp[gt_class] += 1
        else:
            fn[gt_class] += 1

    for class_id in range(num_classes):
        pred_not_matched = np.array([not is_matched(g) for g in pred_geoms])
        fp[class_id] = np.sum(pred_not_matched[pred_classes == class_id])

    return tp, fp, fn


class ObjectDetectionEvaluation(ClassificationEvaluation):
    def __init__(self, class_config: 'ClassConfig', iou_thresh: float = 0.5):
        super().__init__()
        self.class_config = class_config
        self.iou_thresh = iou_thresh

    def compute(self, ground_truth_labels: 'ObjectDetectionLabels',
                prediction_labels: 'ObjectDetectionLabels'):
        self.class_to_eval_item = ObjectDetectionEvaluation.compute_eval_items(
            ground_truth_labels,
            prediction_labels,
            self.class_config,
            iou_thresh=self.iou_thresh)
        self.compute_avg()

    @staticmethod
    def compute_eval_items(
            gt_labels: 'ObjectDetectionLabels',
            pred_labels: 'ObjectDetectionLabels',
            class_config: 'ClassConfig',
            iou_thresh: float = 0.5) -> Dict[int, ClassEvaluationItem]:
        num_classes = len(class_config)
        tps, fps, fns = compute_metrics(gt_labels, pred_labels, num_classes,
                                        iou_thresh)
        class_to_eval_item = {}
        for class_id, (tp, fp, fn) in enumerate(zip(tps, fps, fns)):
            class_name = class_config.get_name(class_id)
            eval_item = ClassEvaluationItem(
                class_id=class_id, class_name=class_name, tp=tp, fp=fp, fn=fn)
            class_to_eval_item[class_id] = eval_item

        return class_to_eval_item
