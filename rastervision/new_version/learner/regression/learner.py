import warnings
warnings.filterwarnings('ignore')
from os.path import join, isfile, isdir
import csv
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision.transforms import (Compose, ToTensor, Resize, ColorJitter,
                                    RandomVerticalFlip, RandomHorizontalFlip)
from PIL import Image

from rastervision.utils.files import (download_if_needed, list_paths,
                                      get_local_path)
from rastervision.new_version.learner.learner import Learner


class ImageRegressionDataset(Dataset):
    def __init__(self, data_dir, labels, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        labels_path = join(data_dir, 'labels.csv')
        with open(labels_path, 'r') as labels_file:
            labels_reader = csv.reader(labels_file, skipinitialspace=True)
            header = next(labels_reader)
            self.output_inds = [header.index(col) for col in labels]
            self.labels = list(labels_reader)[1:]
        self.img_dir = join(data_dir, 'img')

    def __getitem__(self, ind):
        label_row = self.labels[ind]
        img_fn = label_row[0]

        y = [float(label_row[i]) for i in self.output_inds]
        y = torch.tensor(y).float()
        img = Image.open(join(self.img_dir, img_fn))
        if self.transform:
            img = self.transform(img)
        return (img, y)

    def __len__(self):
        return len(self.labels)


class RegressionModel(nn.Module):
    def __init__(self,
                 backbone_arch,
                 out_features,
                 pretrained=True,
                 pos_out_inds=None):
        super().__init__()
        self.backbone = getattr(models, backbone_arch)(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, out_features)
        self.pos_out_inds = pos_out_inds

    def forward(self, x):
        out = self.backbone(x)
        if self.pos_out_inds:
            for ind in self.pos_out_inds:
                out[:, ind] = out[:, ind].exp()
        return out


class RegressionLearner(Learner):
    def build_model(self):
        backbone = self.cfg.model.backbone
        out_features = len(self.cfg.data.labels)
        pos_out_inds = [
            self.cfg.data.labels.index(l) for l in self.cfg.data.pos_labels
        ]
        model = RegressionModel(
            backbone, out_features, pos_out_inds=pos_out_inds)
        return model

    def build_data(self):
        cfg = self.cfg
        batch_sz = cfg.solver.batch_sz
        num_workers = cfg.data.num_workers

        # download and unzip data
        if cfg.data.uri.startswith('s3://') or cfg.data.uri.startswith('/'):
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

        # build datasets -- one per zip file and then merge them into a single dataset
        train_ds = []
        valid_ds = []
        test_ds = []
        for data_dir in data_dirs:
            train_dir = join(data_dir, 'train')
            valid_dir = join(data_dir, 'valid')

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
                        ImageRegressionDataset(
                            train_dir, cfg.data.labels, transform=transform))
                else:
                    train_ds.append(
                        ImageRegressionDataset(
                            train_dir,
                            cfg.data.labels,
                            transform=aug_transform))

            if isdir(valid_dir):
                valid_ds.append(
                    ImageRegressionDataset(
                        valid_dir, cfg.data.labels, transform=transform))
                test_ds.append(
                    ImageRegressionDataset(
                        valid_dir, cfg.data.labels, transform=transform))

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

    def on_overfit_start(self):
        self.on_train_start()

    def on_train_start(self):
        ys = []
        for _, y in self.train_dl:
            ys.append(y)
        y = torch.cat(ys, dim=0)
        self.target_medians = y.median(dim=0).values.to(self.device)

    def build_metric_names(self):
        metric_names = [
            'epoch', 'train_time', 'valid_time', 'train_loss', 'val_loss'
        ]
        for label in self.cfg.data.labels:
            metric_names.extend([
                '{}_abs_error'.format(label),
                '{}_scaled_abs_error'.format(label)
            ])
        return metric_names

    def train_step(self, batch, batch_nb):
        x, y = batch
        out = self.model(x)
        return {'train_loss': F.l1_loss(out, y, reduction='sum')}

    def validate_step(self, batch, batch_nb):
        x, y = batch
        out = self.model(x)
        val_loss = F.l1_loss(out, y, reduction='sum')
        abs_error = torch.abs(out - y).sum(dim=0)
        scaled_abs_error = (
            torch.abs(out - y) / self.target_medians).sum(dim=0)

        metrics = {'val_loss': val_loss}
        for ind, label in enumerate(self.cfg.data.labels):
            metrics['{}_abs_error'.format(label)] = abs_error[ind]
            metrics['{}_scaled_abs_error'.format(label)] = scaled_abs_error[
                ind]

        return metrics

    def plot_xyz(self, ax, x, y, z=None):
        x = x.permute(1, 2, 0)
        if x.shape[2] == 1:
            x = torch.cat([x for _ in range(3)], dim=2)
        ax.imshow(x)

        title = 'true: '
        for _y in y:
            title += '{:.2f} '.format(_y)
        if z is not None:
            title += '\npred: '
            for _z in z:
                title += '{:.2f} '.format(_z)
        ax.set_title(title)
        ax.axis('off')

    def eval_model(self, split):
        super().eval_model(split)

        y, out = self.predict_dataloader(
            self.get_dataloader(split), return_x=False)
        # make scatter plot
        num_labels = len(self.cfg.data.labels)
        ncols = num_labels
        nrows = 1
        fig = plt.figure(
            constrained_layout=True, figsize=(5 * ncols, 5 * nrows))
        grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

        for label_ind, label in enumerate(self.cfg.data.labels):
            ax = fig.add_subplot(grid[label_ind])
            ax.scatter(
                y[:, label_ind], out[:, label_ind], c='blue', alpha=0.02)
            ax.set_title('{} on {} set'.format(label, split))
            ax.set_xlabel('ground truth')
            ax.set_ylabel('predictions')
        scatter_path = join(self.output_dir, '{}_scatter.png'.format(split))
        plt.savefig(scatter_path)

        # make histogram of errors
        fig = plt.figure(
            constrained_layout=True, figsize=(5 * ncols, 5 * nrows))
        grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

        for label_ind, label in enumerate(self.cfg.data.labels):
            ax = fig.add_subplot(grid[label_ind])
            errs = torch.abs(y[:, label_ind] - out[:, label_ind])
            ax.hist(errs, bins=100, density=True)
            ax.set_title('{} on {} set'.format(label, split))
            ax.set_xlabel('prediction error')
        scatter_path = join(self.output_dir, '{}_err_hist.png'.format(split))
        plt.savefig(scatter_path)
