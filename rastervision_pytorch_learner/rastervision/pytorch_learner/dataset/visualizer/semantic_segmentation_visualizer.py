from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from rastervision.pytorch_learner.dataset.visualizer import Visualizer  # NOQA
from rastervision.pytorch_learner.utils import (
    color_to_triple, plot_channel_groups, channel_groups_to_imgs)

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes
    from matplotlib.colors import Colormap


class SemanticSegmentationVisualizer(Visualizer):
    """Plots samples from semantic segmentation Datasets."""

    def plot_xyz(self,
                 axs: Sequence,
                 x: torch.Tensor,
                 y: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 z: Optional[torch.Tensor] = None,
                 plot_title: bool = True) -> None:
        channel_groups = self.get_channel_display_groups(x.shape[1])

        img_axes = axs[:len(channel_groups)]

        # plot image
        imgs = channel_groups_to_imgs(x, channel_groups)
        plot_channel_groups(
            img_axes, imgs, channel_groups, plot_title=plot_title)

        if y is None and z is None:
            return

        # plot labels
        class_colors = self.class_colors
        colors = [
            color_to_triple(c) if isinstance(c, str) else c
            for c in class_colors
        ]
        colors = np.array(colors) / 255.
        cmap = mcolors.ListedColormap(colors)

        if y is not None:
            label_ax: 'Axes' = axs[len(channel_groups)]
            self.plot_gt(label_ax, y, num_classes=len(colors), cmap=cmap)
            if plot_title:
                label_ax.set_title('Ground truth')

        if z is not None:
            pred_ax = axs[-1]
            self.plot_pred(pred_ax, z, num_classes=len(colors), cmap=cmap)
            if plot_title:
                pred_ax.set_title('Predicted labels')

        # add a legend to the rightmost subplot
        class_names = self.class_names
        if class_names:
            legend_items = [
                mpatches.Patch(facecolor=col, edgecolor='black', label=name)
                for col, name in zip(colors, class_names)
            ]
            axs[-1].legend(
                handles=legend_items,
                loc='center left',
                bbox_to_anchor=(1., 0.5))

    def plot_gt(self, ax: 'Axes', y: Union[torch.Tensor, np.ndarray],
                num_classes: int, cmap: 'Colormap', **kwargs):
        ax.imshow(
            y,
            vmin=0,
            vmax=num_classes,
            cmap=cmap,
            interpolation='none',
            **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_pred(self, ax: 'Axes', z: Union[torch.Tensor, np.ndarray],
                  num_classes: int, cmap: 'Colormap', **kwargs):
        if z.ndim == 3:
            z = z.argmax(dim=0)
        self.plot_gt(ax, y=z, num_classes=num_classes, cmap=cmap, **kwargs)

    def get_plot_ncols(self, **kwargs) -> int:
        x = kwargs['x']
        nb_img_channels = x.shape[1]
        ncols = len(self.get_channel_display_groups(nb_img_channels))
        if kwargs.get('y') is not None:
            ncols += 1
        if kwargs.get('z') is not None:
            ncols += 1
        return ncols
