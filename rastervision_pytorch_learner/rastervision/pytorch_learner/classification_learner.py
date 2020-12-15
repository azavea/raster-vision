import warnings
warnings.filterwarnings('ignore')  # noqa

import logging

import torch
import torch.nn as nn
from torchvision import models

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils import (compute_conf_mat_metrics,
                                                compute_conf_mat)

log = logging.getLogger(__name__)


class ClassificationLearner(Learner):
    def build_model(self):
        pretrained = self.cfg.model.pretrained
        model = getattr(
            models, self.cfg.model.get_backbone_str())(pretrained=pretrained)
        in_features = model.fc.in_features
        num_labels = len(self.cfg.data.class_names)
        model.fc = nn.Linear(in_features, num_labels)
        return model

    def build_loss(self):
        loss_weights = self.cfg.solver.class_loss_weights
        if loss_weights is not None:
            loss_weights = torch.tensor(loss_weights, device=self.device)
            loss = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            loss = nn.CrossEntropyLoss()
        return loss

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

    def plot_xyz(self, ax, x, y, z=None):
        if x.shape[2] == 1:
            x = torch.cat([x for _ in range(3)], dim=2)
        ax.imshow(x)
        title = 'true: {}'.format(self.cfg.data.class_names[y])
        if z is not None:
            title += ' / pred: {}'.format(self.cfg.data.class_names[z])
        ax.set_title(title, fontsize=8)
        ax.axis('off')
