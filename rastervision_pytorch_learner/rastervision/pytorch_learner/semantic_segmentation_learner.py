import warnings
warnings.filterwarnings('ignore')  # noqa

from typing import Union, Iterable, Optional

import logging

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use('Agg')  # noqa
import albumentations as A

import torch
from torch import nn
from torchvision import models

from rastervision.pipeline.config import ConfigError
from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils import (
    compute_conf_mat_metrics, compute_conf_mat, color_to_triple, SplitTensor,
    Parallel, AddTensors)
from rastervision.pipeline.file_system import make_dir

log = logging.getLogger(__name__)


class SemanticSegmentationLearner(Learner):
    def build_model(self) -> nn.Module:
        # TODO support FCN option
        pretrained = self.cfg.model.pretrained
        out_classes = len(self.cfg.data.class_names)
        if self.cfg.solver.ignore_last_class:
            out_classes -= 1
        model = models.segmentation.segmentation._segm_resnet(
            'deeplabv3',
            self.cfg.model.get_backbone_str(),
            out_classes,
            False,
            pretrained_backbone=pretrained)

        input_channels = self.cfg.data.img_channels
        old_conv = model.backbone.conv1

        if input_channels == old_conv.in_channels:
            return model

        # these parameters will be the same for the new conv layer
        old_conv_args = {
            'out_channels': old_conv.out_channels,
            'kernel_size': old_conv.kernel_size,
            'stride': old_conv.stride,
            'padding': old_conv.padding,
            'dilation': old_conv.dilation,
            'groups': old_conv.groups,
            'bias': old_conv.bias
        }

        if not pretrained:
            # simply replace the first conv layer with one with the
            # correct number of input channels
            new_conv = nn.Conv2d(in_channels=input_channels, **old_conv_args)
            model.backbone.conv1 = new_conv
            return model

        if input_channels > old_conv.in_channels:
            # insert a new conv layer parallel to the existing one
            # and sum their outputs
            new_conv_channels = input_channels - old_conv.in_channels
            new_conv = nn.Conv2d(
                in_channels=new_conv_channels, **old_conv_args)
            model.backbone.conv1 = nn.Sequential(
                # split input along channel dim
                SplitTensor((old_conv.in_channels, new_conv_channels), dim=1),
                # each split goes to its respective conv layer
                Parallel(old_conv, new_conv),
                # sum the parallel outputs
                AddTensors())
        elif input_channels < old_conv.in_channels:
            model.backbone.conv1 = nn.Conv2d(
                in_channels=input_channels, **old_conv_args)
            model.backbone.conv1.weight.data[:, :input_channels] = \
                old_conv.weight.data[:, :input_channels]
        else:
            raise ConfigError(f'Something went wrong')

        return model

    def build_loss(self):
        args = {}

        loss_weights = self.cfg.solver.class_loss_weights
        if loss_weights is not None:
            loss_weights = torch.tensor(loss_weights, device=self.device)
            args.update({'weight': loss_weights})

        if self.cfg.solver.ignore_last_class:
            num_classes = len(self.cfg.data.class_names)
            args.update({'ignore_index': num_classes - 1})

        loss = nn.CrossEntropyLoss(**args)

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

    def predict(self, x: torch.Tensor, raw_out: bool = False) -> torch.Tensor:
        x = self.to_batch(x).float()
        x = self.to_device(x, self.device)
        with torch.no_grad():
            out = self.model(x)
            out = self.post_forward(out)
            out = out.softmax(dim=1)
            if not raw_out:
                out = self.prob_to_pred(out)
        out = self.to_device(out, 'cpu')
        return out

    def prob_to_pred(self, x):
        return x.argmax(1)

    def plot_batch(self,
                   x: torch.Tensor,
                   y: Union[torch.Tensor, np.ndarray],
                   output_path: str,
                   z: Optional[torch.Tensor] = None,
                   batch_limit: Optional[int] = None) -> None:
        """Plot a whole batch in a grid using plot_xyz.

        Args:
            x: batch of images
            y: ground truth labels
            output_path: local path where to save plot image
            z: optional predicted labels
            batch_limit: optional limit on (rendered) batch size
        """
        batch_sz, c, h, w = x.shape
        batch_sz = min(batch_sz,
                       batch_limit) if batch_limit is not None else batch_sz
        channel_groups = self.cfg.data.channel_display_groups

        nrows = batch_sz
        # one col for each group + 1 for labels + 1 for predictions
        ncols = len(channel_groups) + 1
        if z is not None:
            ncols += 1

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            constrained_layout=True,
            figsize=(3 * ncols, 3 * nrows))

        # (N, c, h, w) --> (N, h, w, c)
        x = x.permute(0, 2, 3, 1)

        # apply transform, if given
        if self.cfg.data.plot_options.transform is not None:
            tf = A.from_dict(self.cfg.data.plot_options.transform)
            x = tf(image=x.numpy())['image']
            x = torch.from_numpy(x)

        for i in range(batch_sz):
            ax = (fig, axes[i])
            if z is None:
                self.plot_xyz(ax, x[i], y[i])
            else:
                self.plot_xyz(ax, x[i], y[i], z=z[i])

        make_dir(output_path, use_dirname=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    def plot_xyz(self,
                 ax: Iterable,
                 x: torch.Tensor,
                 y: Union[torch.Tensor, np.ndarray],
                 z: Optional[torch.Tensor] = None) -> None:

        channel_groups = self.cfg.data.channel_display_groups

        # make subplot titles
        if not isinstance(channel_groups, dict):
            channel_groups = {
                f'Channels: {[*chs]}': chs
                for chs in channel_groups
            }

        fig, ax = ax
        img_axes = ax[:len(channel_groups)]
        label_ax = ax[len(channel_groups)]

        # plot input image(s)
        for (title, chs), ch_ax in zip(channel_groups.items(), img_axes):
            im = x[..., chs]
            if len(chs) == 1:
                im = im.expand(-1, -1, 3)
            elif len(chs) == 2:
                h, w, _ = x.shape
                third_channel = torch.full((h, w, 1), fill_value=.5)
                im = torch.cat((im, third_channel), dim=-1)
            ch_ax.imshow(im)
            ch_ax.set_title(title)
            ch_ax.set_xticks([])
            ch_ax.set_yticks([])

        class_colors = self.cfg.data.class_colors
        colors = [color_to_triple(c) for c in class_colors]
        colors = np.array(colors) / 255.
        cmap = matplotlib.colors.ListedColormap(colors)

        # plot labels
        label_ax.imshow(
            y, vmin=0, vmax=len(colors), cmap=cmap, interpolation='none')
        label_ax.set_title(f'Ground truth labels')
        label_ax.set_xticks([])
        label_ax.set_yticks([])

        # plot predictions
        if z is not None:
            pred_ax = ax[-1]
            pred_ax.imshow(
                z, vmin=0, vmax=len(colors), cmap=cmap, interpolation='none')
            pred_ax.set_title(f'Predicted labels')
            pred_ax.set_xticks([])
            pred_ax.set_yticks([])

        # add a legend to the rightmost subplot
        class_names = self.cfg.data.class_names
        legend_items = [
            mpatches.Patch(facecolor=col, edgecolor='black', label=name)
            for col, name in zip(colors, class_names)
        ]
        ax[-1].legend(
            handles=legend_items,
            loc='center right',
            bbox_to_anchor=(1.8, 0.5))
