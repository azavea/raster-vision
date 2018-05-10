from rastervision.core.evaluation_item import EvaluationItem


class ObjectDetectionEvaluationItem(EvaluationItem):
    def __init__(self, precision, recall, f1, count_error, gt_count=None,
                 class_id=None, class_name=None):
        super().__init__(
            precision, recall, f1, gt_count=gt_count, class_id=class_id,
            class_name=class_name)
        self.count_error = count_error

    def merge(self, other):
        super().merge(other)

        if other.gt_count > 0:
            total_gt_count = self.gt_count + other.gt_count
            self_ratio = self.gt_count / total_gt_count
            other_ratio = other.gt_count / total_gt_count

            def avg(self_val, other_val):
                return self_ratio * self_val + other_ratio * other_val

            self.count_error = avg(self.count_error, other.count_error)
