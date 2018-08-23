import math

from rastervision.core.class_map import ClassMap
from rastervision.core.evaluation import Evaluation
from rastervision.core.evaluation_item import EvaluationItem
from rastervision.core.label_store import LabelStore
from rastervision.label_stores.segmentation_raster_file import (
    SegmentationRasterFile)
from rastervision.raster_sources.image_file import ImageFile
from rastervision.core.box import Box


class SegmentationEvaluation(Evaluation):
    def compute(self, class_map: ClassMap,
                ground_truth_label_store: LabelStore,
                _prediction_label_store: LabelStore) -> None:

        # Construct new prediction label store.  This allows chips to
        # be read out of the prediction raster with the appropriate
        # transformations applied.
        raster_source = ImageFile(None, _prediction_label_store.sink)
        raster_class_map = ground_truth_label_store.raster_class_map
        prediction_label_store = SegmentationRasterFile(
            source=raster_source,
            sink=None,
            class_map=class_map,
            raster_class_map=raster_class_map)

        # Compute the intersection of the extents of the ground truth
        # labels and predicted labels.
        gt_extent = ground_truth_label_store.source.get_extent()
        pr_extent = prediction_label_store.source.get_extent()
        xmin = max(gt_extent.xmin, pr_extent.xmin)
        ymin = max(gt_extent.ymin, pr_extent.ymin)
        xmax = min(gt_extent.xmax, pr_extent.xmax)
        ymax = min(gt_extent.ymax, pr_extent.ymax)
        extent = Box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

        ground_truth = ground_truth_label_store.get_labels(extent)
        predictions = prediction_label_store.get_labels(extent)

        # noqa Definitions of precision, recall, and f1 taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
        evaluation_items = []
        for class_id in class_map.get_keys():
            gt = (ground_truth == class_id)
            pred = (predictions == class_id)
            not_gt = (ground_truth != class_id)
            not_pred = (predictions != class_id)

            true_positives = (gt * pred).sum()
            false_positives = (not_gt * pred).sum()
            false_negatives = (gt * not_pred).sum()

            precision = float(true_positives) / (
                true_positives + false_positives)
            recall = float(true_positives) / (true_positives + false_negatives)
            f1 = 2 * (precision * recall) / (precision + recall)
            count_error = int(false_positives + false_negatives)
            gt_count = int(true_positives.sum())
            class_name = class_map.get_by_id(class_id).name

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

            evaluation_item = EvaluationItem(precision, recall, f1,
                                             count_error, gt_count, class_id,
                                             class_name)
            evaluation_items.append(evaluation_item)

        self.class_to_eval_item = dict(
            zip(class_map.get_keys(), evaluation_items))
        self.compute_avg()

    def compute_avg(self) -> None:
        self.avg_item = EvaluationItem(class_name='average')
        for eval_item in self.class_to_eval_item.values():
            self.avg_item.merge(eval_item)
