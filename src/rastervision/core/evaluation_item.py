class EvaluationItem(object):
    """Evaluation metrics for a single class."""
    def __init__(self, precision, recall, f1, gt_count=None,
                 class_id=None, class_name=None):
        self.precision = precision
        self.recall = recall
        self.f1 = f1

        self.gt_count = gt_count
        self.class_id = class_id
        self.class_name = class_name

    def merge(self, other):
        total_gt_count = self.gt_count + other.gt_count
        self_ratio = 0
        other_ratio = 0
        if total_gt_count > 0:
            self_ratio = self.gt_count / total_gt_count
            other_ratio = other.gt_count / total_gt_count

        def avg(self_val, other_val):
            return self_ratio * self_val + other_ratio * other_val

        self.precision = avg(self.precision, other.precision)
        self.recall = avg(self.recall, other.recall)
        self.f1 = avg(self.f1, other.f1)
        self.gt_count = total_gt_count

    def to_json(self):
        return self.__dict__
