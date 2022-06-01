from typing import Optional, Sequence
import warnings
import logging
from textwrap import wrap

import torch

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils import (
    compute_conf_mat_metrics, compute_conf_mat, plot_channel_groups,
    channel_groups_to_imgs)

warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)


class ClassificationLearner(Learner):
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
