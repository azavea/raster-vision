from rastervision.evaluation import ClassEvaluationItem
from rastervision.evaluation import ClassificationEvaluation


class ObjectDetectionEvaluation(ClassificationEvaluation):
    """Evaluation for object detection.

    Requires TensorFlow Object Detection to be available as it utilizes
    some of its evalution utility functions.
    """

    def __init__(self, class_map):
        super().__init__()
        self.class_map = class_map

    def compute(self, ground_truth_labels, prediction_labels):
        nb_classes = len(self.class_map)
        od_eval = ObjectDetectionEvaluation.compute_od_eval(
            ground_truth_labels, prediction_labels, nb_classes)

        self.class_to_eval_item = ObjectDetectionEvaluation.parse_od_eval(
            od_eval, self.class_map)
        self.compute_avg()

    @staticmethod
    def compute_od_eval(ground_truth_labels, prediction_labels, nb_classes):
        # Lazy import of TFOD
        from object_detection.utils import object_detection_evaluation

        matching_iou_threshold = 0.5
        od_eval = object_detection_evaluation.ObjectDetectionEvaluation(
            nb_classes, matching_iou_threshold=matching_iou_threshold)
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

    @staticmethod
    def parse_od_eval(od_eval, class_map):
        class_to_eval_item = {}

        score_ind = -1
        for class_id in range(1, len(class_map) + 1):
            gt_count = int(od_eval.num_gt_instances_per_class[class_id - 1])
            class_name = class_map.get_by_id(class_id).name

            if gt_count == 0:
                eval_item = ClassEvaluationItem(
                    class_id=class_id, class_name=class_name)
            else:
                # precisions_per_class has an element appended to it for each
                # class_id that has gt_count > 0. This means that the length of
                # precision_per_class can be shorter than the total number of
                # classes in the class_map. Therefore, we use score_ind to index
                # into precisions_per_class instead of simply using class_id - 1.
                score_ind += 1

                # Precisions and recalls across a range of detection thresholds.
                precisions = od_eval.precisions_per_class[score_ind]
                recalls = od_eval.recalls_per_class[score_ind]

                if len(precisions) == 0 or len(recalls) == 0:
                    # No predicted boxes.
                    eval_item = ClassEvaluationItem(
                        precision=None,
                        recall=0,
                        gt_count=gt_count,
                        class_id=class_id,
                        class_name=class_name)
                else:
                    # If we use the lowest detection threshold (ie. use all
                    # detected boxes as defined by score_thresh in the predict
                    # protobuf), that means we use all detected boxes, or the last
                    # element in the precisions array.
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

                    eval_item = ClassEvaluationItem(
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        count_error=norm_count_error,
                        gt_count=gt_count,
                        class_id=class_id,
                        class_name=class_name)

            class_to_eval_item[class_id] = eval_item

        return class_to_eval_item
