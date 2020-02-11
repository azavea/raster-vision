import warnings
warnings.filterwarnings('ignore')  # noqa
from os.path import join, isfile, isdir
import zipfile
import logging

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from albumentations.core.composition import Compose
from albumentations.augmentations.transforms import (
    Blur, RandomRotate90, HorizontalFlip, VerticalFlip, GaussianBlur,
    GaussNoise, RGBShift, ToGray, Resize)

from rastervision2.pipeline.filesystem import (download_if_needed, list_paths,
                                               get_local_path)
from rastervision2.pytorch_learner.learner import Learner
from rastervision2.pytorch_learner.utils import (
    compute_conf_mat_metrics, compute_conf_mat, AlbumentationsDataset)
from rastervision2.pytorch_learner.image_folder import (
    ImageFolder)

log = logging.getLogger(__name__)


class ClassificationLearner(Learner):
    def build_model(self):
        model = getattr(models, self.cfg.model.backbone)(pretrained=True)
        in_features = model.fc.in_features
        num_labels = len(self.cfg.data.class_names)
        model.fc = nn.Linear(in_features, num_labels)
        return model

    def build_data(self):
        cfg = self.cfg
        batch_sz = cfg.solver.batch_sz
        num_workers = cfg.data.num_workers
        label_names = cfg.data.class_names

        # download and unzip data
        if cfg.data.data_format == 'image_folder':
            if cfg.data.uri.startswith('s3://') or cfg.data.uri.startswith(
                    '/'):
                data_uri = cfg.data.uri
            else:
                data_uri = join(cfg.base_uri, cfg.data.uri)

            data_dirs = []
            zip_uris = [data_uri] if data_uri.endswith('.zip') else list_paths(
                data_uri, 'zip')
            for zip_ind, zip_uri in enumerate(zip_uris):
                zip_path = get_local_path(zip_uri, self.data_cache_dir)
                if not isfile(zip_path):
                    zip_path = download_if_needed(zip_uri, self.data_cache_dir)
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    data_dir = join(self.tmp_dir, 'data', str(zip_ind))
                    data_dirs.append(data_dir)
                    zipf.extractall(data_dir)

        transform = Compose(
            [Resize(cfg.data.img_sz, cfg.data.img_sz)])

        augmentors_dict = {
            'Blur': Blur(),
            'RandomRotate90': RandomRotate90(),
            'HorizontalFlip': HorizontalFlip(),
            'VerticalFlip': VerticalFlip(),
            'GaussianBlur': GaussianBlur(),
            'GaussNoise': GaussNoise(),
            'RGBShift': RGBShift(),
            'ToGray': ToGray()
        }
        aug_transforms = [Resize(cfg.data.img_sz, cfg.data.img_sz)]
        for augmentor in cfg.data.augmentors:
            try:
                aug_transforms.append(augmentors_dict[augmentor])
            except KeyError as e:
                log.warning('{0} is an unknown augmentor. Continuing without {0}. \
                    Known augmentors are: {1}'
                            .format(e, list(augmentors_dict.keys())))
        aug_transform = Compose(aug_transforms)

        train_ds, valid_ds, test_ds = [], [], []
        for data_dir in data_dirs:
            train_dir = join(data_dir, 'train')
            valid_dir = join(data_dir, 'valid')

            # build datasets
            if isdir(train_dir):
                if cfg.overfit_mode:
                    train_ds.append(
                        AlbumentationsDataset(
                            ImageFolder(
                                train_dir, classes=label_names),
                            transform=transform))
                else:
                    train_ds.append(
                        AlbumentationsDataset(
                            ImageFolder(
                                train_dir, classes=label_names),
                            transform=aug_transform))

            if isdir(valid_dir):
                valid_ds.append(
                    AlbumentationsDataset(
                        ImageFolder(
                            valid_dir, classes=label_names),
                        transform=transform))
                test_ds.append(
                    AlbumentationsDataset(
                        ImageFolder(
                            valid_dir, classes=label_names),
                        transform=transform))

        train_ds, valid_ds, test_ds = \
            ConcatDataset(train_ds), ConcatDataset(valid_ds), ConcatDataset(test_ds)

        if cfg.overfit_mode:
            train_ds = Subset(train_ds, range(batch_sz))
            valid_ds = train_ds
            test_ds = train_ds
        elif cfg.test_mode:
            train_ds = Subset(train_ds, range(batch_sz))
            valid_ds = Subset(valid_ds, range(batch_sz))
            test_ds = Subset(test_ds, range(batch_sz))

        train_dl = DataLoader(
            train_ds,
            shuffle=True,
            batch_size=batch_sz,
            num_workers=num_workers,
            pin_memory=True)
        valid_dl = DataLoader(
            valid_ds,
            shuffle=True,
            batch_size=batch_sz,
            num_workers=num_workers,
            pin_memory=True)
        test_dl = DataLoader(
            test_ds,
            shuffle=True,
            batch_size=batch_sz,
            num_workers=num_workers,
            pin_memory=True)

        self.train_ds, self.valid_ds, self.test_ds = (train_ds, valid_ds,
                                                      test_ds)
        self.train_dl, self.valid_dl, self.test_dl = (train_dl, valid_dl,
                                                      test_dl)

    def train_step(self, batch, batch_nb):
        x, y = batch
        out = self.model(x)
        return {'train_loss': F.cross_entropy(out, y, reduction='sum')}

    def validate_step(self, batch, batch_nb):
        x, y = batch
        out = self.model(x)
        val_loss = F.cross_entropy(out, y, reduction='sum')

        num_labels = len(self.cfg.data.class_names)
        out = self.post_forward(out)
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
        return x.argmax(-1)

    def plot_xyz(self, ax, x, y, z=None):
        x = x.permute(1, 2, 0)
        if x.shape[2] == 1:
            x = torch.cat([x for _ in range(3)], dim=2)
        ax.imshow(x)
        title = 'true: {}'.format(self.cfg.data.class_names[y])
        if z is not None:
            title += ' / pred: {}'.format(self.cfg.data.class_names[z])
        ax.set_title(title, fontsize=8)
        ax.axis('off')
