from typing import Optional, Sequence
import warnings

import logging

import torch
import torch.nn as nn
from torchvision import models
from textwrap import wrap

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils import (
    compute_conf_mat_metrics, compute_conf_mat, adjust_conv_channels,
    plot_channel_groups, channel_groups_to_imgs)
from rastervision.pipeline.config import ConfigError

warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)


class ClassificationLearner(Learner):
    def build_model(self):
        pretrained = self.cfg.model.pretrained
        num_classes = len(self.cfg.data.class_names)
        backbone_name = self.cfg.model.get_backbone_str()
        in_channels = self.cfg.data.img_channels
        if in_channels is None:
            log.warn('DataConfig.img_channels is None. Defaulting to 3.')
            in_channels = 3

        model = getattr(models, backbone_name)(pretrained=pretrained)

        if in_channels != 3:
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

    def get_plot_ncols(self, **kwargs) -> int:
        ncols = len(self.cfg.data.plot_options.channel_display_groups) + 1
        return ncols

    def plot_xyz(self,
                 axs: Sequence,
                 x: torch.Tensor,
                 y: int,
                 z: Optional[int] = None) -> None:

        channel_groups = self.cfg.data.plot_options.channel_display_groups

        img_axes = axs[:-1]
        label_ax = axs[-1]

        # plot image
        imgs = channel_groups_to_imgs(x, channel_groups)
        plot_channel_groups(img_axes, imgs, channel_groups)

        # plot label
        class_names = self.cfg.data.class_names
        class_names = ['-\n-'.join(wrap(c, width=8)) for c in class_names]
        if z is None:
            # just display the class name as text
            class_name = class_names[y]
            label_ax.text(
                .5,
                .5,
                class_name,
                ha='center',
                va='center',
                fontdict={
                    'size': 24,
                    'family': 'sans-serif'
                })
            label_ax.set_xlim((0, 1))
            label_ax.set_ylim((0, 1))
            label_ax.axis('off')
        else:
            # display predicted class probabilities as a horizontal bar plot
            # legend: green = ground truth, dark-red = wrong prediction,
            # light-gray = other. In case predicted class matches ground truth,
            # only one bar will be green and the others will be light-gray.
            class_probabilities = z.softmax(dim=-1)
            class_index_pred = z.argmax(dim=-1)
            class_index_gt = y
            bar_colors = ['lightgray'] * len(z)
            if class_index_pred == class_index_gt:
                bar_colors[class_index_pred] = 'green'
            else:
                bar_colors[class_index_pred] = 'darkred'
                bar_colors[class_index_gt] = 'green'
            label_ax.barh(
                y=class_names,
                width=class_probabilities,
                color=bar_colors,
                edgecolor='black')
            label_ax.set_xlim((0, 1))
            label_ax.xaxis.grid(linestyle='--', alpha=1)
            label_ax.set_xlabel('Probability')
            label_ax.set_title('Prediction')
