from typing import Optional, Sequence
import warnings
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from textwrap import wrap

import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils.utils import (
    adjust_conv_channels, plot_channel_groups, channel_groups_to_imgs)
from rastervision.pipeline.config import ConfigError

warnings.filterwarnings('ignore')
matplotlib.use('Agg')


class RegressionModel(nn.Module):
    def __init__(self,
                 backbone_arch,
                 out_features,
                 pretrained=True,
                 pos_out_inds=None,
                 prob_out_inds=None):
        super().__init__()
        self.backbone = getattr(models, backbone_arch)(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, out_features)
        self.pos_out_inds = pos_out_inds
        self.prob_out_inds = prob_out_inds

    def forward(self, x):
        out = self.backbone(x)
        if self.pos_out_inds:
            for ind in self.pos_out_inds:
                out[:, ind] = out[:, ind].exp()
        if self.prob_out_inds:
            for ind in self.prob_out_inds:
                out[:, ind] = out[:, ind].sigmoid()
        return out


class RegressionLearner(Learner):
    def build_model(self):
        pretrained = self.cfg.model.pretrained
        backbone_name = self.cfg.model.get_backbone_str()
        in_channels = self.cfg.data.img_channels
        out_features = len(self.cfg.data.class_names)
        pos_out_inds = [
            self.cfg.data.class_names.index(class_name)
            for class_name in self.cfg.data.pos_class_names
        ]
        prob_out_inds = [
            self.cfg.data.class_names.index(class_name)
            for class_name in self.cfg.data.prob_class_names
        ]
        model = RegressionModel(
            backbone_name,
            out_features,
            pretrained=pretrained,
            pos_out_inds=pos_out_inds,
            prob_out_inds=prob_out_inds)

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
            model.backbone.conv1 = adjust_conv_channels(
                old_conv=model.backbone.conv1,
                in_channels=in_channels,
                pretrained=pretrained)

        return model

    def on_overfit_start(self):
        self.on_train_start()

    def on_train_start(self):
        ys = []
        for _, y in self.train_dl:
            ys.append(y)
        y = torch.cat(ys, dim=0)
        self.target_medians = y.median(dim=0).values.to(self.device)

    def build_metric_names(self):
        metric_names = [
            'epoch', 'train_time', 'valid_time', 'train_loss', 'val_loss'
        ]
        for label in self.cfg.data.class_names:
            metric_names.extend([
                '{}_abs_error'.format(label),
                '{}_scaled_abs_error'.format(label)
            ])
        return metric_names

    def train_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        return {'train_loss': F.l1_loss(out, y, reduction='sum')}

    def validate_step(self, batch, batch_nb):
        x, y = batch
        out = self.post_forward(self.model(x))
        val_loss = F.l1_loss(out, y, reduction='sum')
        abs_error = torch.abs(out - y).sum(dim=0)
        scaled_abs_error = (
            torch.abs(out - y) / self.target_medians).sum(dim=0)

        metrics = {'val_loss': val_loss}
        for ind, label in enumerate(self.cfg.data.class_names):
            metrics['{}_abs_error'.format(label)] = abs_error[ind]
            metrics['{}_scaled_abs_error'.format(label)] = scaled_abs_error[
                ind]

        return metrics

    def prob_to_pred(self, x):
        return x

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
            # display targets as a horizontal bar plot
            bars_gt = label_ax.barh(
                y=class_names, width=y, color='lightgray', edgecolor='black')
            # show values on the end of bars
            label_ax.bar_label(bars_gt, fmt='%.3f', padding=3)

            label_ax.set_title('Ground truth')
        else:
            # display targets and predictions as a grouped horizontal bar plot
            bar_thickness = 0.35
            y_tick_locs = np.arange(len(class_names))
            bars_gt = label_ax.barh(
                y=y_tick_locs + bar_thickness / 2,
                width=y,
                height=bar_thickness,
                color='lightgray',
                edgecolor='black',
                label='true')
            bars_pred = label_ax.barh(
                y=y_tick_locs - bar_thickness / 2,
                width=z,
                height=bar_thickness,
                color=plt.get_cmap('tab10')(0),
                edgecolor='black',
                label='pred')
            # show values on the end of bars
            label_ax.bar_label(bars_gt, fmt='%.3f', padding=3)
            label_ax.bar_label(bars_pred, fmt='%.3f', padding=3)

            label_ax.set_yticks(ticks=y_tick_locs, labels=class_names)
            label_ax.legend(
                ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0))

        label_ax.xaxis.grid(linestyle='--', alpha=1)
        label_ax.set_xlabel('Target value')
        label_ax.spines['right'].set_visible(False)
        label_ax.get_yaxis().tick_left()

    def eval_model(self, split):
        super().eval_model(split)

        y, out = self.predict_dataloader(
            self.get_dataloader(split), return_x=False, raw_out=False)

        max_scatter_points = self.cfg.data.plot_options.max_scatter_points
        if y.shape[0] > max_scatter_points:
            scatter_inds = torch.randperm(
                y.shape[0], dtype=torch.long)[0:max_scatter_points]
        else:
            scatter_inds = torch.arange(0, y.shape[0], dtype=torch.long)

        # make scatter plot
        num_labels = len(self.cfg.data.class_names)
        ncols = num_labels
        nrows = 1
        fig = plt.figure(
            constrained_layout=True, figsize=(5 * ncols, 5 * nrows))
        grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

        for label_ind, label in enumerate(self.cfg.data.class_names):
            ax = fig.add_subplot(grid[label_ind])
            ax.scatter(
                y[scatter_inds, label_ind],
                out[scatter_inds, label_ind],
                c='blue',
                alpha=0.1)
            ax.set_title('{} on {} set'.format(label, split))
            ax.set_xlabel('ground truth')
            ax.set_ylabel('predictions')
        scatter_path = join(self.output_dir, '{}_scatter.png'.format(split))
        plt.savefig(scatter_path)
        print('done scatter')

        # make histogram of errors
        fig = plt.figure(
            constrained_layout=True, figsize=(5 * ncols, 5 * nrows))
        grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

        hist_bins = self.cfg.data.plot_options.hist_bins
        for label_ind, label in enumerate(self.cfg.data.class_names):
            ax = fig.add_subplot(grid[label_ind])
            errs = torch.abs(y[:, label_ind] - out[:, label_ind]).tolist()
            ax.hist(errs, bins=hist_bins)
            ax.set_title('{} on {} set'.format(label, split))
            ax.set_xlabel('prediction error')
        hist_path = join(self.output_dir, '{}_err_hist.png'.format(split))
        plt.savefig(hist_path)
        print('done hist')
