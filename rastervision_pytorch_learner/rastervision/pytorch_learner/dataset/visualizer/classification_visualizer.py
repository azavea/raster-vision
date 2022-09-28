from typing import (Sequence, Optional)
from textwrap import wrap

import torch

from rastervision.pytorch_learner.dataset.visualizer import Visualizer  # NOQA
from rastervision.pytorch_learner.utils import (plot_channel_groups,
                                                channel_groups_to_imgs)


class ClassificationVisualizer(Visualizer):
    """Plots samples from image classification Datasets."""

    def plot_xyz(self,
                 axs: Sequence,
                 x: torch.Tensor,
                 y: int,
                 z: Optional[int] = None) -> None:
        channel_groups = self.get_channel_display_groups(x.shape[1])

        img_axes = axs[:-1]
        label_ax = axs[-1]

        # plot image
        imgs = channel_groups_to_imgs(x, channel_groups)
        plot_channel_groups(img_axes, imgs, channel_groups)

        # plot label
        class_names = self.class_names
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

    def get_plot_ncols(self, **kwargs) -> int:
        x = kwargs['x']
        nb_img_channels = x.shape[1]
        ncols = len(self.get_channel_display_groups(nb_img_channels)) + 1
        return ncols
