import warnings

warnings.filterwarnings('ignore')  # noqa

import logging

import torch
import torch.nn as nn
from torchvision import models

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils import (
    compute_conf_mat_metrics, compute_conf_mat, adjust_conv_channels)
from rastervision.pipeline.config import ConfigError

log = logging.getLogger(__name__)


class ClassificationLearner(Learner):
    def build_model(self):
        pretrained = self.cfg.model.pretrained
        num_classes = len(self.cfg.data.class_names)
        backbone_name = self.cfg.model.get_backbone_str()

        model = getattr(models, backbone_name)(pretrained=pretrained)

        if self.cfg.data.img_channels != 3:
            if not backbone_name.startswith('resnet'):
                raise ConfigError(
                    'All TorchVision backbones do not provide the same API '
                    'for accessing the first conv layer. '
                    'Therefore, conv layer modification to support '
                    'arbitrary input channels is only supported for resnet '
                    'backbones. To use other backbones, it is recommended to '
                    'fork the TorchVision repo, define factory functions or '
                    'subclasses that perform the necessary modifications, and '
                    'then use the external model functionality to import it '
                    'into Raster Vision. See spacenet_rio.py for an example '
                    'of how to import external models. Alternatively, you can '
                    'override this function.')
            model.conv1 = adjust_conv_channels(
                old_conv=model.conv1,
                in_channels=self.cfg.data.img_channels,
                pretrained=pretrained)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

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
