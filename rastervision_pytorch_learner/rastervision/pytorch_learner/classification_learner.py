import warnings
import logging

import torch.distributed as dist

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils import (
    compute_conf_mat_metrics, compute_conf_mat, aggregate_metrics)
from rastervision.pytorch_learner.dataset.visualizer import (
    ClassificationVisualizer)

warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)


class ClassificationLearner(Learner):
    def get_visualizer_class(self):
        return ClassificationVisualizer

    def train_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        return {'train_loss': self.loss(out, y)}

    def validate_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        val_loss = self.loss(out, y)

        num_labels = len(self.cfg.data.class_names)
        out = self.prob_to_pred(out)
        conf_mat = compute_conf_mat(out, y, num_labels)

        return {'val_loss': val_loss, 'conf_mat': conf_mat}

    def validate_end(self, outputs):
        metrics = aggregate_metrics(outputs, exclude_keys={'conf_mat'})
        conf_mat = sum([o['conf_mat'] for o in outputs])

        if self.is_ddp_process:
            metrics = self.reduce_distributed_metrics(metrics)
            dist.reduce(conf_mat, dst=0, op=dist.ReduceOp.SUM)
            if not self.is_ddp_master:
                return metrics

        ignored_idx = self.cfg.solver.ignore_class_index
        if ignored_idx is not None and ignored_idx < 0:
            ignored_idx += self.cfg.data.num_classes

        class_names = self.cfg.data.class_names
        conf_mat_metrics = compute_conf_mat_metrics(
            conf_mat, class_names, ignore_idx=ignored_idx)

        metrics.update(conf_mat_metrics)
        return metrics

    def prob_to_pred(self, x):
        return x.argmax(-1)
