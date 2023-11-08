from typing import TYPE_CHECKING, Optional, Sequence
from textwrap import wrap

import torch
import numpy as np
import matplotlib.pyplot as plt

from rastervision.pytorch_learner.dataset.visualizer import Visualizer  # NOQA
from rastervision.pytorch_learner.utils import (plot_channel_groups,
                                                channel_groups_to_imgs)

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes


class RegressionVisualizer(Visualizer):
    """Plots samples from image regression Datasets."""

    def plot_xyz(self,
                 axs: Sequence,
                 x: torch.Tensor,
                 y: int,
                 z: Optional[int] = None,
                 plot_title: bool = True) -> None:
        channel_groups = self.get_channel_display_groups(x.shape[1])

        img_axes = axs[:-1]
        label_ax: 'Axes' = axs[-1]

        # plot image
        imgs = channel_groups_to_imgs(x, channel_groups)
        plot_channel_groups(
            img_axes, imgs, channel_groups, plot_title=plot_title)

        # plot label
        class_names = self.class_names
        class_names = ['-\n-'.join(wrap(c, width=8)) for c in class_names]

        if y is not None and z is None:
            self.plot_gt(label_ax, class_names, y)
            if plot_title:
                label_ax.set_title('Ground truth')
        elif z is not None:
            self.plot_pred(label_ax, class_names, z, y=y)

    def plot_gt(self, ax: 'Axes', class_names: Sequence[str], y: torch.Tensor):
        """Plot targets as a horizontal bar plot with values at the tips."""
        bars_gt = ax.barh(
            y=class_names, width=y, color='lightgray', edgecolor='black')
        # show values on the end of bars
        ax.bar_label(bars_gt, fmt='%.3f', padding=3)

        ax.xaxis.grid(linestyle='--', alpha=1)
        ax.set_xlabel('Value')
        ax.spines['right'].set_visible(False)
        ax.get_yaxis().tick_left()

    def plot_pred(self,
                  ax: 'Axes',
                  class_names: Sequence[str],
                  z: torch.Tensor,
                  y: Optional[torch.Tensor] = None):
        """Plot targets and predictions as a grouped horizontal bar plot."""
        # display targets and predictions as a grouped horizontal bar plot
        bar_thickness = 0.35 if y is not None else 0.70
        y_tick_locs = np.arange(len(class_names))
        if y is not None:
            bars_gt = ax.barh(
                y=y_tick_locs + bar_thickness / 2,
                width=y,
                height=bar_thickness,
                color='lightgray',
                edgecolor='black',
                label='true')
            # show values on the end of bars
            ax.bar_label(bars_gt, fmt='%.3f', padding=3)

        bars_pred = ax.barh(
            y=y_tick_locs - bar_thickness / 2,
            width=z,
            height=bar_thickness,
            color=plt.get_cmap('tab10')(0),
            edgecolor='black',
            label='pred')
        # show values on the end of bars
        ax.bar_label(bars_pred, fmt='%.3f', padding=3)

        ax.set_yticks(ticks=y_tick_locs, labels=class_names)
        ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0))

        ax.xaxis.grid(linestyle='--', alpha=1)
        ax.set_xlabel('Value')
        ax.spines['right'].set_visible(False)
        ax.get_yaxis().tick_left()

    def get_plot_ncols(self, **kwargs) -> int:
        x = kwargs['x']
        nb_img_channels = x.shape[1]
        ncols = len(self.get_channel_display_groups(nb_img_channels)) + 1
        return ncols
