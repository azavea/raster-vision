import warnings
warnings.filterwarnings('ignore')  # noqa
from os.path import join, isdir
import logging
from typing import Optional

import torch
from torchvision import models
import torch.nn as nn
from torch.utils.data import ConcatDataset

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.learner_config import LearnerConfig
from rastervision.pytorch_learner.utils import (
    compute_conf_mat_metrics, compute_conf_mat, AlbumentationsDataset)
from rastervision.pytorch_learner.image_folder import (ImageFolder)
from rastervision.pytorch_learner.classification_learner_config import (
    ClassificationDataFormat)

log = logging.getLogger(__name__)


class ClassificationLearner(Learner):
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
            loss_weights = torch.tensor(loss_weights, device=self.device)
            self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def build_model(self):
        pretrained = self.cfg.model.pretrained
        model = getattr(
            models, self.cfg.model.get_backbone_str())(pretrained=pretrained)
        in_features = model.fc.in_features
        num_labels = len(self.cfg.data.class_names)
        model.fc = nn.Linear(in_features, num_labels)
        return model

    def _get_datasets(self, uri):
        cfg = self.cfg
        class_names = cfg.data.class_names

        if cfg.data.data_format == ClassificationDataFormat.image_folder:
            data_dirs = self.unzip_data(uri)

        transform, aug_transform = self.get_data_transforms()

        train_ds, valid_ds, test_ds = [], [], []
        for data_dir in data_dirs:
            train_dir = join(data_dir, 'train')
            valid_dir = join(data_dir, 'valid')

            if isdir(train_dir):
                if cfg.overfit_mode:
                    train_ds.append(
                        AlbumentationsDataset(
                            ImageFolder(train_dir, classes=class_names),
                            transform=transform))
                else:
                    train_ds.append(
                        AlbumentationsDataset(
                            ImageFolder(train_dir, classes=class_names),
                            transform=aug_transform))

            if isdir(valid_dir):
                valid_ds.append(
                    AlbumentationsDataset(
                        ImageFolder(valid_dir, classes=class_names),
                        transform=transform))
                test_ds.append(
                    AlbumentationsDataset(
                        ImageFolder(valid_dir, classes=class_names),
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
        out = self.prob_to_pred(out)
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

    def prob_to_pred(self, x):
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
