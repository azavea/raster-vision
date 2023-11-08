from typing import TYPE_CHECKING, Optional, Sequence
from textwrap import wrap

import torch

from rastervision.pytorch_learner.dataset.visualizer import Visualizer  # NOQA
from rastervision.pytorch_learner.utils import (plot_channel_groups,
                                                channel_groups_to_imgs)

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes


class ClassificationVisualizer(Visualizer):
    """Plots samples from image classification Datasets."""

    def plot_xyz(self,
                 axs: Sequence['Axes'],
                 x: torch.Tensor,
                 y: Optional[int] = None,
                 z: Optional[int] = None,
                 plot_title: bool = True) -> None:
        channel_groups = self.get_channel_display_groups(x.shape[1])

        img_axes = axs[:-1]
        label_ax = axs[-1]

        # plot image
        imgs = channel_groups_to_imgs(x, channel_groups)
        plot_channel_groups(
            img_axes, imgs, channel_groups, plot_title=plot_title)

        # plot label
        class_names = self.class_names
        class_names = ['-\n-'.join(wrap(c, width=16)) for c in class_names]
        if y is not None and z is None:
            self.plot_gt(label_ax, class_names, y)
        elif z is not None:
            self.plot_pred(label_ax, class_names, z, y=y)
            if plot_title:
                label_ax.set_title('Prediction')

    def plot_gt(self, ax: 'Axes', class_names: Sequence[str], y: torch.Tensor):
        """Display ground truth class names as text."""
        class_name = class_names[y]
        ax.text(
            x=.5,
            y=.5,
            s=class_name,
            ha='center',
            va='center',
            fontdict={
                'size': 20,
                'family': 'sans-serif'
            })
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.axis('off')

    def plot_pred(self,
                  ax: 'Axes',
                  class_names: Sequence[str],
                  z: torch.Tensor,
                  y: Optional[torch.Tensor] = None):
        """Plot predictions.

        Plots predicted class probabilities as a horizontal bar plot. If ground
        truth, y, is provided, the bar colors represent: green = ground truth,
        dark-red = wrong prediction, light-gray = other. In case predicted
        class matches ground truth, only one bar will be green and the others
        will be light-gray.
        """
        class_probabilities = z.softmax(dim=-1)
        class_index_pred = z.argmax(dim=-1)
        bar_colors = ['lightgray'] * len(z)
        if y is not None:
            class_index_gt = y
            if class_index_pred == class_index_gt:
                bar_colors[class_index_pred] = 'green'
            else:
                bar_colors[class_index_pred] = 'darkred'
                bar_colors[class_index_gt] = 'green'
        ax.barh(
            y=class_names,
            width=class_probabilities,
            color=bar_colors,
            edgecolor='black')
        ax.set_xlim((0, 1))
        ax.xaxis.grid(linestyle='--', alpha=1)
        ax.set_xlabel('Probability')

    def get_plot_ncols(self, **kwargs) -> int:
        x = kwargs['x']
        nb_img_channels = x.shape[1]
        ncols = len(self.get_channel_display_groups(nb_img_channels)) + 1
        return ncols
