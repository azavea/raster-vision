import warnings
warnings.filterwarnings('ignore')
from os.path import join, isfile, isdir
import zipfile

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.transforms import (Compose, ToTensor, Resize, ColorJitter,
                                    RandomVerticalFlip, RandomHorizontalFlip)

from rastervision.backend.torch_utils.chip_classification.folder import (
    ImageFolder)
from rastervision.utils.files import (download_if_needed, list_paths,
                                      get_local_path)
from rastervision.new_version.learner.learner import Learner
from rastervision.new_version.learner.metrics import (compute_conf_mat_metrics,
                                                      compute_conf_mat)


class ClassificationLearner(Learner):
    def build_model(self):
        model = getattr(models, self.cfg.model.backbone)(pretrained=True)
        in_features = model.fc.in_features
        num_labels = len(self.cfg.data.labels)
        model.fc = nn.Linear(in_features, num_labels)
        return model

    def build_data(self):
        cfg = self.cfg
        batch_sz = cfg.solver.batch_sz
        num_workers = cfg.data.num_workers
        label_names = cfg.data.labels

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

        train_ds, valid_ds, test_ds = [], [], []
        for data_dir in data_dirs:
            train_dir = join(data_dir, 'train')
            valid_dir = join(data_dir, 'valid')

            # build datasets
            transform = Compose(
                [Resize((cfg.data.img_sz, cfg.data.img_sz)),
                 ToTensor()])
            aug_transform = Compose([
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ColorJitter(0.1, 0.1, 0.1, 0.1),
                Resize((cfg.data.img_sz, cfg.data.img_sz)),
                ToTensor()
            ])

            if isdir(train_dir):
                if cfg.overfit_mode:
                    train_ds.append(
                        ImageFolder(
                            train_dir,
                            transform=transform,
                            classes=label_names))
                else:
                    train_ds.append(
                        ImageFolder(
                            train_dir,
                            transform=aug_transform,
                            classes=label_names))

            if isdir(valid_dir):
                valid_ds.append(
                    ImageFolder(
                        valid_dir, transform=transform, classes=label_names))
                test_ds.append(
                    ImageFolder(
                        valid_dir, transform=transform, classes=label_names))

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

        # build dataloaders
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_sz, num_workers=num_workers, pin_memory=True) \
            if train_ds else None
        valid_dl = DataLoader(valid_ds, shuffle=True, batch_size=batch_sz, num_workers=num_workers, pin_memory=True) \
            if valid_ds else None
        test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_sz, num_workers=num_workers, pin_memory=True) \
            if test_ds else None

        self.train_ds, self.valid_ds, self.test_ds = train_ds, valid_ds, test_ds
        self.train_dl, self.valid_dl, self.test_dl = train_dl, valid_dl, test_dl

    def train_step(self, batch, batch_nb):
        x, y = batch
        out = self.model(x)
        return {'train_loss': F.cross_entropy(out, y, reduction='sum')}

    def validate_step(self, batch, batch_nb):
        x, y = batch
        out = self.model(x)
        val_loss = F.cross_entropy(out, y, reduction='sum')

        num_labels = len(self.cfg.data.labels)
        out = self.post_forward(out)
        conf_mat = compute_conf_mat(out, y, num_labels)

        return {'val_loss': val_loss, 'conf_mat': conf_mat}

    def validate_end(self, outputs, num_samples):
        conf_mat = sum([o['conf_mat'] for o in outputs])
        val_loss = torch.stack([o['val_loss']
                                for o in outputs]).sum() / num_samples
        conf_mat_metrics = compute_conf_mat_metrics(conf_mat,
                                                    self.cfg.data.labels)

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
        title = 'true: {}'.format(self.cfg.data.labels[y])
        if z is not None:
            title += ' / pred: {}'.format(self.cfg.data.labels[z])
        ax.set_title(title, fontsize=8)
        ax.axis('off')
