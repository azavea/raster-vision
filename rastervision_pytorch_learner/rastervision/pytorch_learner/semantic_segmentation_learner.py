import warnings
warnings.filterwarnings('ignore')  # noqa

from typing import Union, IO, Callable, List, Iterable, Optional, Tuple
from os.path import join, isdir
from pathlib import Path

import logging

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use('Agg')  # noqa
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, ConcatDataset
from torchvision import models

from rastervision.pipeline.config import ConfigError
from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.learner_config import LearnerConfig
from rastervision.pytorch_learner.utils import (
    compute_conf_mat_metrics, compute_conf_mat, color_to_triple, SplitTensor,
    Parallel, AddTensors)
from rastervision.pipeline.file_system import make_dir

log = logging.getLogger(__name__)


def load_np_normalized(path: Union[IO, str, Path]) -> np.ndarray:
    arr = np.load(path)
    dtype = arr.dtype
    if np.issubdtype(dtype, np.unsignedinteger):
        max_val = np.iinfo(dtype).max
        arr = arr.astype(np.float32) / max_val
    return arr


class SemanticSegmentationDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 img_fmt: str = 'png',
                 label_fmt: str = 'png',
                 transform: Callable = None):

        self.data_dir = Path(data_dir)
        img_dir = self.data_dir / 'img'
        label_dir = self.data_dir / 'labels'

        # collect image and label paths
        self.img_paths = list(img_dir.glob(f'*.{img_fmt}'))
        self.label_paths = [
            label_dir / f'{p.stem}.{label_fmt}' for p in self.img_paths
        ]

        # choose image loading method based on format
        if img_fmt.lower() in ('npy', 'npz'):
            self.img_load_fn = load_np_normalized
        else:
            self.img_load_fn = lambda path: np.array(Image.open(path)) / 255.

        # choose label loading method based on format
        if label_fmt.lower() in ('npy', 'npz'):
            self.label_load_fn = np.load
        else:
            self.label_load_fn = lambda path: np.array(Image.open(path))

        self.transform = transform

    def __getitem__(self,
                    ind: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:

        img_path = self.img_paths[ind]
        label_path = self.label_paths[ind]

        x = self.img_load_fn(img_path)
        y = self.label_load_fn(label_path)

        if x.ndim == 2:
            # (h, w) --> (h, w, 1)
            x = x[..., np.newaxis]

        if self.transform is not None:
            out = self.transform(image=x, mask=y)
            x = out['image']
            y = out['mask']

        x = torch.from_numpy(x).permute(2, 0, 1).float()
        y = torch.from_numpy(y).long()

        return (x, y)

    def __len__(self):
        return len(self.img_paths)


class SemanticSegmentationLearner(Learner):
    def __init__(self,
                 cfg: LearnerConfig,
                 tmp_dir: str,
                 model_path: Optional[str] = None):
        """Constructor.

        Args:
            cfg: configuration
            tmp_dir: root of temp dirs
            model_path: a local path to model weights. If provided, the model is loaded
                and it is assumed that this Learner will be used for prediction only.
        """
        super().__init__(cfg, tmp_dir, model_path)

        loss_weights = self.cfg.solver.class_loss_weights
        if loss_weights is not None:
            loss_weights = torch.Tensor(loss_weights, device=self.device)
            self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def build_model(self) -> nn.Module:
        # TODO support FCN option
        pretrained = self.cfg.model.pretrained
        model = models.segmentation.segmentation._segm_resnet(
            'deeplabv3',
            self.cfg.model.get_backbone_str(),
            len(self.cfg.data.class_names),
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
        else:
            raise ConfigError(
                (f'Fewer input channels ({input_channels}) than what'
                 f'the pretrained model expects ({old_conv.in_channels})'))

        return model

    def _get_datasets(self, uri: Union[str, List[str]]):
        cfg = self.cfg

        data_dirs = self.unzip_data(uri)
        transform, aug_transform = self.get_data_transforms()
        img_fmt, label_fmt = cfg.data.img_format, cfg.data.label_format

        train_ds, valid_ds, test_ds = [], [], []
        for data_dir in data_dirs:
            train_dir = join(data_dir, 'train')
            valid_dir = join(data_dir, 'valid')

            if isdir(train_dir):
                tf = transform if cfg.overfit_mode else aug_transform
                ds = SemanticSegmentationDataset(
                    train_dir,
                    img_fmt=img_fmt,
                    label_fmt=label_fmt,
                    transform=tf)
                train_ds.append(ds)

            if isdir(valid_dir):
                valid_ds.append(
                    SemanticSegmentationDataset(
                        valid_dir,
                        img_fmt=img_fmt,
                        label_fmt=label_fmt,
                        transform=transform))
                test_ds.append(
                    SemanticSegmentationDataset(
                        valid_dir,
                        img_fmt=img_fmt,
                        label_fmt=label_fmt,
                        transform=transform))

        train_ds, valid_ds, test_ds = \
            ConcatDataset(train_ds), ConcatDataset(valid_ds), ConcatDataset(test_ds)

        return train_ds, valid_ds, test_ds

    def train_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        return {'train_loss': self.loss_fn(out, y)}

    def validate_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        val_loss = self.loss_fn(out, y)

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
        return x['out']

    def prob_to_pred(self, x):
        return x.argmax(1)

    def plot_batch(self,
                   x: torch.Tensor,
                   y: Union[torch.Tensor, np.ndarray],
                   output_path: str,
                   z: Optional[torch.Tensor] = None) -> None:
        """Plot a whole batch in a grid using plot_xyz.

        Args:
            x: batch of images
            y: ground truth labels
            output_path: local path where to save plot image
            z: optional predicted labels
        """
        batch_sz, c, h, w = x.shape
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

        # (c, h, w) --> (h, w, c)
        x = x.permute(1, 2, 0)

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
