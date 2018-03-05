import json

from object_detection.utils import object_detection_evaluation

from rv2.core.evaluation import Evaluation
from rv2.utils.files import str_to_file


class EvaluationItem(object):
    def __init__(self, precision, recall, f1, count_error,
                 gt_count=None, class_id=None, class_name=None):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.count_error = count_error

        self.gt_count = gt_count
        self.class_id = class_id
        self.class_name = class_name

    def merge(self, other):
        total_gt_count = self.gt_count + other.gt_count
        self_ratio = self.gt_count / total_gt_count
        other_ratio = other.gt_count / total_gt_count

        def avg(self_val, other_val):
            return self_ratio * self_val + other_ratio * other_val

        self.precision = avg(self.precision, other.precision)
        self.recall = avg(self.recall, other.recall)
        self.f1 = avg(self.f1, other.f1)
        self.count_error = avg(self.count_error, other.count_error)
        self.gt_count = total_gt_count

    def to_json(self):
        return self.__dict__


def compute_od_eval(ground_truth_annotations, prediction_annotations):
    nb_gt_classes = len(set(ground_truth_annotations.get_classes()))
    matching_iou_threshold = 0.5
    od_eval = object_detection_evaluation.ObjectDetectionEvaluation(
        nb_gt_classes, matching_iou_threshold=matching_iou_threshold)
    image_key = 'image'
    od_eval.add_single_ground_truth_image_info(
        image_key, ground_truth_annotations.get_npboxes(),
        ground_truth_annotations.get_classes() - 1)
    od_eval.add_single_detected_image_info(
        image_key, prediction_annotations.get_npboxes(),
        prediction_annotations.get_scores(),
        prediction_annotations.get_classes() - 1)
    od_eval.evaluate()
    return od_eval


def parse_od_eval(od_eval, label_map):
    eval_result = od_eval.get_eval_result()
    class_to_eval_item = {}
    for class_id in range(1, len(label_map) + 1):
        class_name = label_map.get_by_id(class_id).name

        precisions = eval_result.precisions[class_id - 1]
        recalls = eval_result.recalls[class_id - 1]
        # Get precision and recall assuming all predicted boxes are used.
        precision = float(precisions[-1])
        recall = float(recalls[-1])
        f1 = (2 * precision * recall) / (precision + recall)

        gt_count = int(od_eval.num_gt_instances_per_class[class_id - 1])
        pred_count = len(recalls)
        count_error = pred_count - gt_count
        norm_count_error = count_error / gt_count

        eval_item = EvaluationItem(
            precision, recall, f1, norm_count_error, gt_count=gt_count,
            class_id=class_id, class_name=class_name)
        class_to_eval_item[class_id] = eval_item

    return class_to_eval_item


class ObjectDetectionEvaluation(Evaluation):
    def __init__(self):
        self.clear()

    def clear(self):
        self.class_to_eval_item = {}
        self.avg_item = None

    def get_by_id(self, class_id):
        return self.class_to_eval_item[class_id]

    def compute(self, label_map, ground_truth_annotation_source,
                prediction_annotation_source):
        gt_annotations = ground_truth_annotation_source.get_all_annotations()
        pred_annotations = prediction_annotation_source.get_all_annotations()

        od_eval = compute_od_eval(gt_annotations, pred_annotations)
        self.class_to_eval_item = parse_od_eval(od_eval, label_map)

        self.compute_avg()

    def compute_avg(self):
        self.avg_item = EvaluationItem(0, 0, 0, 0, gt_count=0,
                                       class_name='average')
        for eval_item in self.class_to_eval_item.values():
            self.avg_item.merge(eval_item)

    def merge(self, evaluation):
        if len(self.class_to_eval_item) == 0:
            self.class_to_eval_item = evaluation.class_to_eval_item
        else:
            for class_id, other_eval_item in evaluation.class_to_eval_item.items():
                self.get_by_id(class_id).merge(other_eval_item)

        self.compute_avg()

    def to_json(self):
        json_rep = []
        for eval_item in self.class_to_eval_item.values():
            json_rep.append(eval_item.to_json())
        json_rep.append(self.avg_item.to_json())
        return json_rep

    def save(self, output_uri):
        json_str = json.dumps(self.to_json(), indent=4)
        str_to_file(json_str, output_uri)
