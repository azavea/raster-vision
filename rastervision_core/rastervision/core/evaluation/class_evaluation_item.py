import numpy as np

from rastervision.core.evaluation import EvaluationItem


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
                 class_name=None,
                 conf_mat=None):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.count_error = count_error
        # Ground truth count of elements (boxes for object detection, pixels
        # for segmentation, cells for classification).
        self.gt_count = gt_count
        self.conf_mat = conf_mat
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

        if other.conf_mat is not None:
            if self.class_name == 'average':
                if self.conf_mat is None:
                    # Make first row all zeros so that the array indices
                    # correspond to valid class ids (ie. >= 1).
                    self.conf_mat = np.concatenate(
                        [
                            np.zeros_like(other.conf_mat)[np.newaxis, :],
                            np.array(other.conf_mat)[np.newaxis, :]
                        ],
                        axis=0)
                else:
                    self.conf_mat = np.concatenate(
                        [self.conf_mat, other.conf_mat[np.newaxis, :]], axis=0)
            else:
                self.conf_mat += other.conf_mat

    def to_json(self):
        new_dict = {}
        for k, v in self.__dict__.items():
            new_dict[k] = v.tolist() if isinstance(v, np.ndarray) else v
        if new_dict['conf_mat'] is None:
            del new_dict['conf_mat']
        return new_dict

    def __repr__(self):
        return str(self.to_json())
