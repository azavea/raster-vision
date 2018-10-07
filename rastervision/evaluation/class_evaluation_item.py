from rastervision.evaluation import EvaluationItem


class ClassEvaluationItem(EvaluationItem):
    """Evaluation metrics for a single class (or average over classes).

    None is used for values that are undefined because they involve a division
    by zero (eg. precision when there are no predictions).
    """

    def __init__(self,
                 precision=None,
                 recall=None,
                 f1=None,
                 count_error=None,
                 gt_count=0,
                 class_id=None,
                 class_name=None):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.count_error = count_error
        # Ground truth count of elements (boxes for object detection, pixels
        # for segmentation, cells for classification).
        self.gt_count = gt_count
        self.class_id = class_id
        self.class_name = class_name

    def merge(self, other):
        """Merges another item from a different scene into this one.

        This is used to average metrics over scenes. Merges by taking a
        weighted average (by gt_count) of the metrics.
        """
        if other.gt_count > 0:
            total_gt_count = self.gt_count + other.gt_count
            self_ratio = self.gt_count / total_gt_count
            other_ratio = other.gt_count / total_gt_count

            def weighted_avg(self_val, other_val):
                if self_val is None and other_val is None:
                    return 0.0
                # Handle a single None value by setting them to zero.
                return (self_ratio * (self_val or 0) +
                        other_ratio * (other_val or 0))

            self.precision = weighted_avg(self.precision, other.precision)
            self.recall = weighted_avg(self.recall, other.recall)
            self.f1 = weighted_avg(self.f1, other.f1)
            self.count_error = weighted_avg(self.count_error,
                                            other.count_error)
            self.gt_count = total_gt_count

    def to_json(self):
        return self.__dict__

    def __repr__(self):
        return str(self.to_json())
