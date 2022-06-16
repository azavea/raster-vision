from typing import Sequence, Union, Optional
import warnings

import logging

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

import torch
from torch.nn import functional as F

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils import (
    compute_conf_mat_metrics, compute_conf_mat, color_to_triple,
    plot_channel_groups, channel_groups_to_imgs)

warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)


class SemanticSegmentationLearner(Learner):
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

    def numpy_predict(self, x: np.ndarray,
                      raw_out: bool = False) -> np.ndarray:
        _, h, w, _ = x.shape
        transform, _ = self.cfg.data.get_data_transforms()
        x = self.normalize_input(x)
        x = self.to_batch(x)
        x = np.stack([transform(image=img)['image'] for img in x])
        x = torch.from_numpy(x)
        x = x.permute((0, 3, 1, 2))
        out = self.predict(x, raw_out=True)
        out = F.interpolate(
            out, size=(h, w), mode='bilinear', align_corners=False)
        if not raw_out:
            out = self.prob_to_pred(out)
        return self.output_to_numpy(out)

    def prob_to_pred(self, x):
        return x.argmax(1)

    def get_plot_ncols(self, **kwargs) -> int:
        ncols = len(self.cfg.data.plot_options.channel_display_groups) + 1
        z = kwargs.get('z')
        if z is not None:
            ncols += 1
        return ncols

    def plot_xyz(self,
                 axs: Sequence,
                 x: torch.Tensor,
                 y: Union[torch.Tensor, np.ndarray],
                 z: Optional[torch.Tensor] = None) -> None:

        channel_groups = self.cfg.data.plot_options.channel_display_groups

        img_axes = axs[:len(channel_groups)]
        label_ax = axs[len(channel_groups)]

        # plot image
        imgs = channel_groups_to_imgs(x, channel_groups)
        plot_channel_groups(img_axes, imgs, channel_groups)

        # plot labels
        class_colors = self.cfg.data.class_colors
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
        class_names = self.cfg.data.class_names
        legend_items = [
            mpatches.Patch(facecolor=col, edgecolor='black', label=name)
            for col, name in zip(colors, class_names)
        ]
        axs[-1].legend(
            handles=legend_items,
            loc='center right',
            bbox_to_anchor=(1.8, 0.5))
