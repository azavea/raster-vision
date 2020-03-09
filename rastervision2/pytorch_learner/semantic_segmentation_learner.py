import warnings
warnings.filterwarnings('ignore')  # noqa
from os.path import join, isdir, basename
import logging
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')  # noqa
import torch
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F
from torchvision import models
from PIL import Image

from rastervision2.pytorch_learner.learner import Learner
from rastervision2.pytorch_learner.utils import (compute_conf_mat_metrics,
                                                 compute_conf_mat)
from rastervision2.core.data.utils import color_to_triple

log = logging.getLogger(__name__)


class SemanticSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.img_paths = glob.glob(join(data_dir, 'img', '*.png'))
        self.transform = transform

    def __getitem__(self, ind):
        img_path = self.img_paths[ind]
        label_path = join(self.data_dir, 'labels', basename(img_path))
        x = Image.open(img_path)
        y = Image.open(label_path)

        x = np.array(x)
        y = np.array(y)
        if self.transform is not None:
            out = self.transform(image=x, mask=y)
            x = out['image']
            y = out['mask']

        x = torch.tensor(x).permute(2, 0, 1).float() / 255.0
        y = torch.tensor(y).long()

        return (x, y)

    def __len__(self):
        return len(self.img_paths)


class SemanticSegmentationLearner(Learner):
    def build_model(self):
        model = models.segmentation.segmentation._segm_resnet(
            'deeplabv3',
            self.cfg.model.backbone,
            len(self.cfg.data.class_names),
            False,
            pretrained_backbone=True)
        return model

    def get_datasets(self):
        cfg = self.cfg

        if cfg.data.data_format == 'default':
            data_dirs = self.unzip_data()

        transform, aug_transform = self.get_data_transforms()

        train_ds, valid_ds, test_ds = [], [], []
        for data_dir in data_dirs:
            train_dir = join(data_dir, 'train')
            valid_dir = join(data_dir, 'valid')

            if isdir(train_dir):
                if cfg.overfit_mode:
                    train_ds.append(
                        SemanticSegmentationDataset(
                            train_dir, transform=transform))
                else:
                    train_ds.append(
                        SemanticSegmentationDataset(
                            train_dir, transform=aug_transform))

            if isdir(valid_dir):
                valid_ds.append(
                    SemanticSegmentationDataset(
                        valid_dir, transform=transform))
                test_ds.append(
                    SemanticSegmentationDataset(
                        valid_dir, transform=transform))

        train_ds, valid_ds, test_ds = \
            ConcatDataset(train_ds), ConcatDataset(valid_ds), ConcatDataset(test_ds)

        return train_ds, valid_ds, test_ds

    def train_step(self, batch, batch_nb):
        x, y = batch
        out = self.post_forward(self.model(x))
        return {'train_loss': F.cross_entropy(out, y)}

    def validate_step(self, batch, batch_nb):
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
        if x.shape[2] == 1:
            x = torch.cat([x for _ in range(3)], dim=2)
        ax.imshow(x)
        ax.axis('off')

        labels = z if z is not None else y
        colors = [color_to_triple(c) for c in self.cfg.data.class_colors]
        colors = [tuple([_c / 255 for _c in c]) for c in colors]
        cmap = matplotlib.colors.ListedColormap(colors)
        labels = labels.numpy()
        ax.imshow(labels, alpha=0.4, vmin=0, vmax=len(colors), cmap=cmap)
