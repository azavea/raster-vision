import json

import numpy as np

from object_detection.utils import object_detection_evaluation

from rastervision.core.evaluation import Evaluation
from rastervision.utils.files import str_to_file
from rastervision.evaluation_items.object_detection_evaluation_item import (
    ObjectDetectionEvaluationItem)


def compute_od_eval(ground_truth_labels, prediction_labels):
    nb_gt_classes = len(set(ground_truth_labels.get_class_ids()))
    matching_iou_threshold = 0.5
    od_eval = object_detection_evaluation.ObjectDetectionEvaluation(
        nb_gt_classes, matching_iou_threshold=matching_iou_threshold)
    image_key = 'image'
    od_eval.add_single_ground_truth_image_info(
        image_key, ground_truth_labels.get_npboxes(),
        ground_truth_labels.get_class_ids() - 1)
    od_eval.add_single_detected_image_info(
        image_key, prediction_labels.get_npboxes(),
        prediction_labels.get_scores(),
        prediction_labels.get_class_ids() - 1)
    od_eval.evaluate()
    return od_eval


def parse_od_eval(od_eval, class_map):
    class_to_eval_item = {}
    for class_id in range(1, len(class_map) + 1):
        class_name = class_map.get_by_id(class_id).name
        gt_count = int(od_eval.num_gt_instances_per_class[class_id - 1])

        # If there are predictions for this class.
        if len(od_eval.precisions_per_class[class_id - 1]) > 0:
            precisions = od_eval.precisions_per_class[class_id - 1]
            recalls = od_eval.recalls_per_class[class_id - 1]
            # Get precision and recall assuming all predicted boxes are used.
            precision = float(precisions[-1])
            recall = float(recalls[-1])
            f1 = 0.
            if precision + recall != 0.0:
                f1 = (2 * precision * recall) / (precision + recall)

            pred_count = len(recalls)
            count_error = pred_count - gt_count
            norm_count_error = None
            if gt_count > 0:
                norm_count_error = count_error / gt_count

            eval_item = ObjectDetectionEvaluationItem(
                precision, recall, f1, norm_count_error, gt_count=gt_count,
                class_id=class_id, class_name=class_name)
        else:
            eval_item = ObjectDetectionEvaluationItem(
                0, 0, 0, 1.0, gt_count=gt_count, class_id=class_id,
                class_name=class_name)

        class_to_eval_item[class_id] = eval_item

    return class_to_eval_item


class ObjectDetectionEvaluation(Evaluation):
    def compute(self, class_map, ground_truth_label_store,
                prediction_label_store):
        gt_labels = ground_truth_label_store.get_all_labels()
        pred_labels = prediction_label_store.get_all_labels()

        od_eval = compute_od_eval(gt_labels, pred_labels)
        self.class_to_eval_item = parse_od_eval(od_eval, class_map)

        self.compute_avg()

    def compute_avg(self):
        self.avg_item = ObjectDetectionEvaluationItem(
            0, 0, 0, 0, gt_count=0, class_name='average')
        for eval_item in self.class_to_eval_item.values():
            self.avg_item.merge(eval_item)
