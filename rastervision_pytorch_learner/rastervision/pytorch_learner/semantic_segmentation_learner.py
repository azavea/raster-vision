import warnings
warnings.filterwarnings('ignore')  # noqa

from typing import Union, IO, Callable, List
from os.path import join, isdir
from pathlib import Path

import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')  # noqa
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
from torchvision import models

from rastervision.pipeline.config import ConfigError
from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils import (
    compute_conf_mat_metrics, compute_conf_mat, color_to_triple,
    SplitTensor, Parallel, AddTensors)

log = logging.getLogger(__name__)


def load_np_normalized(path: Union[IO, str, Path]):
    arr = np.load(path)
    dtype = arr.dtype
    if np.issubdtype(dtype, np.unsignedinteger):
        max_val = np.iinfo(dtype).max
        arr = arr.astype(np.float32) / max_val
    return arr


class SemanticSegmentationDataset(Dataset):
    def __init__(self, data_dir: str, img_fmt: str = 'png', label_fmt: str = 'png', transform: Callable = None):

        self.data_dir = Path(data_dir)
        img_dir   = self.data_dir / 'img'
        label_dir = self.data_dir / 'labels'
        
        # collect image and label paths
        self.img_paths = list(img_dir.glob(f'*.{img_fmt}'))
        self.label_paths = [
            label_dir/f'{p.stem}.{label_fmt}' for p in self.img_paths]

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
        import IPython; IPython.embed(); exit(1)

    def __getitem__(self, ind: int):

        img_path   = self.img_paths[ind]
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
    def build_model(self):
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
        old_conv_channels = old_conv.in_channels

        if input_channels == old_conv_channels:
            return model

        if not pretrained:
            new_conv = torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=old_conv.bias
            )
            model.backbone.conv1 = new_conv
            return model

        if input_channels > old_conv_channels:
            new_conv_channels = input_channels - old_conv_channels
            new_conv = torch.nn.Conv2d(
                in_channels=new_conv_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=old_conv.bias
            )
            model.backbone.conv1 = nn.Sequential(
                SplitTensor((old_conv_channels, new_conv_channels), dim=1),
                Parallel(old_conv, new_conv),
                AddTensors()
            )
        else:
            raise ConfigError(
                f'Fewer input channels ({input_channels}) than what the pretrained model expects ({old_conv_channels})')

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
                    train_dir, img_fmt=img_fmt, label_fmt=label_fmt, 
                    transform=tf)
                train_ds.append(ds)

            if isdir(valid_dir):
                valid_ds.append(
                    SemanticSegmentationDataset(
                        valid_dir, img_fmt=img_fmt, label_fmt=label_fmt, 
                        transform=transform))
                test_ds.append(
                    SemanticSegmentationDataset(
                        valid_dir, img_fmt=img_fmt, label_fmt=label_fmt, 
                        transform=transform))

        train_ds, valid_ds, test_ds = \
            ConcatDataset(train_ds), ConcatDataset(valid_ds), ConcatDataset(test_ds)

        return train_ds, valid_ds, test_ds

    def train_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        return {'train_loss': F.cross_entropy(out, y)}

    def validate_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        val_loss = F.cross_entropy(out, y)

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

    def plot_xyz(self, ax, x, y, z=None):
        x = x.permute(1, 2, 0)

        # h, w, c = x.shape
        # channel_groups = self.cfg.data.channel_display_groups
        # # if not specified, just use the first 3 channels
        # if not channel_groups:
        #     channel_groups = [ (0, 1, 2)[: min(3, c)] ]

        # for chs in channel_groups:
        #     if len(chs) == 3:
        #         im = x[..., chs]
        #     elif len(chs) == 1:
        #         im = x[..., chs].expand(-1, -1, 3)
        #     elif len(chs) == 2:
        #         third_channel = torch.full((h, w, 1), .5)
        #         im = torch.cat((x, third_channel), dim=-1)
        if x.shape[2] == 1:
            x = torch.cat([x for _ in range(3)], dim=2)
        ax.imshow(x)
        ax.axis('off')

        labels = z if z is not None else y
        colors = [color_to_triple(c) for c in self.cfg.data.class_colors]
        colors = [tuple([_c / 255 for _c in c]) for c in colors]
        cmap = matplotlib.colors.ListedColormap(colors)
        labels = labels.numpy()
        ax.imshow(
            labels,
            alpha=0.4,
            vmin=0,
            vmax=len(colors),
            cmap=cmap,
            interpolation='none')
