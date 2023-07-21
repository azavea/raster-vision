"""Defines ``ClassEvaluationItem``."""

from typing import Optional

import numpy as np

from rastervision.core.evaluation import EvaluationItem


class ClassEvaluationItem(EvaluationItem):
    """A wrapper around a binary (2x2) confusion matrix of the form

    .. line-block::
       [``TN`` ``FP``]
       [``FN`` ``TP``]

    where ``TN`` need not necessarily be available.

    Exposes evaluation metrics computed from the confusion matrix as
    properties.

    Attributes:
        class_id (int): Class ID.
        class_name (str): Class name.
        conf_mat (np.ndarray): Confusion matrix: ``[[TN, FP], [FN, TP]]``.
        extra_info (dict): Arbitrary extra key-value pairs that will be
            included in the dict returned by ``to_json()``.
    """

    def __init__(self,
                 class_id: int,
                 class_name: str,
                 tp: int,
                 fp: int,
                 fn: int,
                 tn: Optional[int] = None,
                 **kwargs):
        """Constructor.

        Args:
            class_id (int): Class ID.
            class_name (str): Class name.
            tp (int): True positive count.
            fp (int): False positive count.
            fn (int): False negative count.
            tn (Optional[int], optional): True negative count.
                Defaults to None.
            **kwargs: Additional data can be provided as keyword arguments.
                These will be included in the dict returned by ``to_json()``.
        """
        self.class_id = class_id
        self.class_name = class_name
        if tn is None:
            tn = -1
        self.conf_mat = np.array([[tn, fp], [fn, tp]])
        self.extra_info = kwargs

    @classmethod
    def from_multiclass_conf_mat(cls, conf_mat: np.ndarray, class_id: int,
                                 class_name: str,
                                 **kwargs) -> 'ClassEvaluationItem':
        """Construct from a multi-class confusion matrix and a target class ID.

        Args:
            conf_mat (np.ndarray): A multi-class confusion matrix.
            class_id (int): The ID of the target class.
            class_name (str): The name of the target class.
            **kwargs: Extra args for :meth:`.__init__`.

        Returns:
            ClassEvaluationItem: ClassEvaluationItem for target class.
        """
        tp = conf_mat[class_id, class_id]
        fp = conf_mat[:, class_id].sum() - tp
        fn = conf_mat[class_id, :].sum() - tp
        tn = conf_mat.sum() - tp - fp - fn
        item = cls(
            class_id=class_id,
            class_name=class_name,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            **kwargs)
        return item

    def merge(self, other: 'ClassEvaluationItem') -> None:
        """Merge with another ``ClassEvaluationItem``.

        This is accomplished by summing the confusion matrices.
        """
        if self.class_id != other.class_id:
            raise ValueError(
                'Cannot merge evaluation items for different classes.')
        self.conf_mat += other.conf_mat

    @property
    def gt_count(self) -> int:
        """Positive ground-truth count."""
        return self.conf_mat[1, :].sum()

    @property
    def pred_count(self) -> int:
        """Positive prediction count."""
        return self.conf_mat[:, 1].sum()

    @property
    def true_pos(self) -> int:
        """True positive count."""
        return self.conf_mat[1, 1]

    @property
    def true_neg(self) -> Optional[int]:
        """True negative count.

        Returns:
            Optional[int]: Count as int if available. Otherwise, None.
        """
        tn = self.conf_mat[0, 0]
        if tn < 0:
            return None
        return tn

    @property
    def false_pos(self) -> int:
        """False positive count."""
        return self.conf_mat[0, 1]

    @property
    def false_neg(self) -> int:
        """False negative count."""
        return self.conf_mat[1, 0]

    @property
    def recall(self) -> float:
        """``TP / (TP + FN)``"""
        tp = self.true_pos
        fn = self.false_neg
        return float(tp) / (tp + fn)

    @property
    def sensitivity(self) -> float:
        """Equivalent to ``recall``."""
        return self.recall

    @property
    def specificity(self) -> Optional[float]:
        """``TN / (TN + FP)``"""
        if self.true_neg is None:
            return None
        tn = self.true_neg
        fp = self.false_pos
        return float(tn) / (tn + fp)

    @property
    def precision(self) -> float:
        """``TP / (TP + FP)``"""
        tp = self.true_pos
        fp = self.false_pos
        return float(tp) / (tp + fp)

    @property
    def f1(self) -> float:
        """F1 score = ``2 * (precision * recall) / (precision + recall)``"""
        precision = self.precision
        recall = self.recall
        return 2 * (precision * recall) / (precision + recall)

    def to_json(self) -> dict:
        """Serialize to a dict."""
        out = {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'gt_count': self.gt_count,
            'pred_count': self.pred_count,
            'count_error': abs(self.gt_count - self.pred_count),
            'relative_frequency': self.gt_count / self.conf_mat.sum(),
            'metrics': {
                'recall': self.recall,
                'precision': self.precision,
                'f1': self.f1,
                'sensitivity': self.sensitivity,
                'specificity': self.specificity,
            }
        }
        if self.true_neg is None:
            del out['relative_frequency']
            out['true_pos'] = self.true_pos
            out['false_pos'] = self.false_pos
            out['false_neg'] = self.false_neg
        else:
            cm = self.conf_mat
            cm_frac = cm / cm.sum()
            out['conf_mat'] = cm.tolist()
            out['conf_mat_frac'] = cm_frac.tolist()
            [[TN, FP], [FN, TP]] = cm
            out['conf_mat_dict'] = dict(TN=TN, FP=FP, FN=FN, TP=TP)
            [[TN, FP], [FN, TP]] = cm_frac
            out['conf_mat_frac_dict'] = dict(TN=TN, FP=FP, FN=FN, TP=TP)

        out.update(self.extra_info)
        return out
