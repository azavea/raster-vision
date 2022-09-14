from typing import (Sequence, Optional, Union)

import torch
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from rastervision.pytorch_learner.dataset.visualizer import Visualizer  # NOQA
from rastervision.pytorch_learner.utils import (
    color_to_triple, plot_channel_groups, channel_groups_to_imgs)


class SemanticSegmentationVisualizer(Visualizer):
    """Plots samples from semantic segmentation Datasets."""

    def plot_xyz(self,
                 axs: Sequence,
                 x: torch.Tensor,
                 y: Union[torch.Tensor, np.ndarray],
                 z: Optional[torch.Tensor] = None) -> None:
        channel_groups = self.get_channel_display_groups(x.shape[1])

        img_axes = axs[:len(channel_groups)]
        label_ax = axs[len(channel_groups)]

        # plot image
        imgs = channel_groups_to_imgs(x, channel_groups)
        plot_channel_groups(img_axes, imgs, channel_groups)

        # plot labels
        class_colors = self.class_colors
        colors = [
            color_to_triple(c) if isinstance(c, str) else c
            for c in class_colors
        ]
        colors = np.array(colors) / 255.
        cmap = mcolors.ListedColormap(colors)

        label_ax.imshow(
            y, vmin=0, vmax=len(colors), cmap=cmap, interpolation='none')
        label_ax.set_title(f'Ground truth')
        label_ax.set_xticks([])
        label_ax.set_yticks([])

        # plot predictions
        if z is not None:
            pred_ax = axs[-1]
            preds = z.argmax(dim=0)
            pred_ax.imshow(
                preds,
                vmin=0,
                vmax=len(colors),
                cmap=cmap,
                interpolation='none')
            pred_ax.set_title(f'Predicted labels')
            pred_ax.set_xticks([])
            pred_ax.set_yticks([])

        # add a legend to the rightmost subplot
        class_names = self.class_names
        if class_names:
            legend_items = [
                mpatches.Patch(facecolor=col, edgecolor='black', label=name)
                for col, name in zip(colors, class_names)
            ]
            axs[-1].legend(
                handles=legend_items,
                loc='center right',
                bbox_to_anchor=(1.8, 0.5))

    def get_plot_ncols(self, **kwargs) -> int:
        x = kwargs['x']
        nb_img_channels = x.shape[1]
        ncols = len(self.get_channel_display_groups(nb_img_channels)) + 1
        z = kwargs.get('z')
        if z is not None:
            ncols += 1
        return ncols
