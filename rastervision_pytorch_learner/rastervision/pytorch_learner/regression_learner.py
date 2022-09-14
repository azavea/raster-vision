from typing import TYPE_CHECKING, Optional
import warnings
from os.path import join
import logging

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn.functional as F

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.dataset.visualizer import (
    RegressionVisualizer)

if TYPE_CHECKING:
    import torch.nn as nn

warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)


class RegressionLearner(Learner):
    def get_visualizer_class(self):
        return RegressionVisualizer

    def build_model(self, model_def_path: Optional[str] = None) -> 'nn.Module':
        """Override to pass class_names, pos_class_names, and prob_class_names.
        """
        cfg = self.cfg
        class_names = cfg.data.class_names
        pos_class_names = cfg.data.pos_class_names
        prob_class_names = cfg.data.prob_class_names
        model = cfg.model.build(
            num_classes=cfg.data.num_classes,
            in_channels=cfg.data.img_channels,
            save_dir=self.modules_dir,
            hubconf_dir=model_def_path,
            class_names=class_names,
            pos_class_names=pos_class_names,
            prob_class_names=prob_class_names)
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

    def eval_model(self, split):
        super().eval_model(split)

        y, out = self.predict_dataloader(
            self.get_dataloader(split), return_format='yz', raw_out=False)

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
