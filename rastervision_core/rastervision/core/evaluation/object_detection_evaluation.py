from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import geopandas as gpd

from rastervision.core.evaluation import (ClassificationEvaluation,
                                          ClassEvaluationItem)

if TYPE_CHECKING:
    from rastervision.core.data import ObjectDetectionLabels
    from rastervision.core.data.class_config import ClassConfig


def compute_metrics(
        gt_labels: 'ObjectDetectionLabels',
        pred_labels: 'ObjectDetectionLabels',
        num_classes: int,
        iou_thresh: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class true positives, false positives, and false negatives.

    Does the following:

    1.  Spatially join ground truth (GT) boxes with predicted boxes.
    2.  Compute intersection-overo-union (IoU) for each matched box-pair.
    3.  Filter matches by ``iou_thresh``.
    4.  For each GT box >1 matches, keep only the max-IoU one.
    5.  For each pred box >1 matches, keep only the max-IoU one.
    6.  For each class, c, compute:

        a.  True positives (TP) := #matches where GT class ID == c and
            pred class ID == c
        b.  False positives := #preds where (class ID == c) minus TP
        c.  False negatives := #GT where (class ID == c) minus TP

    """
    gt_geoms = [b.to_shapely() for b in gt_labels.get_boxes()]
    gt_classes = gt_labels.get_class_ids()
    pred_geoms = [b.to_shapely() for b in pred_labels.get_boxes()]
    pred_classes = pred_labels.get_class_ids()

    gt_df = gpd.GeoDataFrame(
        dict(class_id=gt_classes, id=range(len(gt_geoms))), geometry=gt_geoms)
    pred_df = gpd.GeoDataFrame(
        dict(class_id=pred_classes, id=range(len(pred_geoms))),
        geometry=pred_geoms)

    gt_df.loc[:, '_geometry'] = gt_df.geometry
    pred_df.loc[:, '_geometry'] = pred_df.geometry

    match_df: gpd.GeoDataFrame = gt_df.sjoin(
        pred_df,
        how='inner',
        predicate='intersects',
        lsuffix='gt',
        rsuffix='pred')

    intersection = match_df['_geometry_gt'].intersection(
        match_df['_geometry_pred'])
    union = match_df['_geometry_gt'].union(match_df['_geometry_pred'])
    match_df.loc[:, 'iou'] = (intersection.area / union.area)
    match_df = match_df.loc[match_df['iou'] > iou_thresh]
    match_df = match_df.sort_values('iou').drop_duplicates(
        ['id_gt'], keep='last')
    match_df = match_df.sort_values('iou').drop_duplicates(
        ['id_pred'], keep='last')

    tp = np.zeros((num_classes, ))
    fp = np.zeros((num_classes, ))
    fn = np.zeros((num_classes, ))

    for class_id in range(num_classes):
        tp[class_id] = sum((match_df['class_id_gt'] == class_id)
                           & (match_df['class_id_pred'] == class_id))
        fp[class_id] = sum(pred_df['class_id'] == class_id) - tp[class_id]
        fn[class_id] = sum(gt_df['class_id'] == class_id) - tp[class_id]

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
