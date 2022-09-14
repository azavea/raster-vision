import warnings
import logging

import torch

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils import (compute_conf_mat_metrics,
                                                compute_conf_mat)
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

    def validate_end(self, outputs, num_samples):
        conf_mat = sum([o['conf_mat'] for o in outputs])
        val_loss = torch.stack([o['val_loss']
                                for o in outputs]).sum() / num_samples
        conf_mat_metrics = compute_conf_mat_metrics(conf_mat,
                                                    self.cfg.data.class_names)

        metrics = {'val_loss': val_loss.item()}
        metrics.update(conf_mat_metrics)

        return metrics

    def prob_to_pred(self, x):
        return x.argmax(-1)
