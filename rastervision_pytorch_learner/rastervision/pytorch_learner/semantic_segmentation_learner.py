from typing import Optional, Tuple
import warnings

import logging

import torch
from torch.nn import functional as F

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils import (compute_conf_mat_metrics,
                                                compute_conf_mat)
from rastervision.pytorch_learner.dataset.visualizer import (
    SemanticSegmentationVisualizer)

warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)


class SemanticSegmentationLearner(Learner):
    def get_visualizer_class(self):
        return SemanticSegmentationVisualizer

    def train_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        return {'train_loss': self.loss(out, y)}

    def validate_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        val_loss = self.loss(out, y)

        num_labels = len(self.cfg.data.class_names)
        y = y.view(-1)
        out = self.prob_to_pred(out).view(-1)
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

    def post_forward(self, x):
        if isinstance(x, dict):
            return x['out']
        return x

    def predict(self,
                x: torch.Tensor,
                raw_out: bool = False,
                out_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if out_shape is None:
            out_shape = x.shape[-2:]

        x = self.to_batch(x).float()
        x = self.to_device(x, self.device)

        with torch.inference_mode():
            out = self.model(x)
            out = self.post_forward(out)
            out = out.softmax(dim=1)

        # ensure correct output shape
        if out.shape[-2:] != out_shape:
            out = F.interpolate(
                out, size=out_shape, mode='bilinear', align_corners=False)

        if not raw_out:
            out = self.prob_to_pred(out)
        out = self.to_device(out, 'cpu')

        return out

    def prob_to_pred(self, x):
        return x.argmax(1)
