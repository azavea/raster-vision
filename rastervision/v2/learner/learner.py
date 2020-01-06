from os.path import join, isfile
import csv
import warnings
warnings.filterwarnings('ignore')  # noqa
import time
import datetime
from abc import ABC, abstractmethod
import shutil
import os
import math

import click
import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR
from rastervision.v2.core.filesystem import (sync_to_dir, json_to_file, file_to_json,
                                      make_dir, zipdir, download_if_needed,
                                      sync_from_dir, get_local_path, unzip)

from rastervision.v2.core.config import build_config
from rastervision.v2.learner.learner_config import LearnerConfig


class Learner(ABC):
    def __init__(self, cfg: LearnerConfig, tmp_dir, model_path=None):
        self.cfg = cfg
        self.tmp_dir = tmp_dir

        torch_cache_dir = '/opt/data/torch-cache'
        os.environ['TORCH_HOME'] = torch_cache_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_cache_dir = '/opt/data/data-cache'
        make_dir(self.data_cache_dir)

        self.model = self.build_model()
        self.model.to(self.device)

        if model_path is not None:
            if isfile(model_path):
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device))
            else:
                raise Exception(
                    'Model could not be found at {}'.format(model_path))
            self.model.eval()
        else:
            print(self.cfg)

            self.train_ds = None
            self.train_dl = None
            self.valid_ds = None
            self.valid_dl = None
            self.test_ds = None
            self.test_dl = None

            if cfg.output_uri.startswith('s3://'):
                self.output_dir = get_local_path(cfg.output_uri, tmp_dir)
                make_dir(self.output_dir, force_empty=True)
                if not cfg.overfit_mode:
                    self.sync_from_cloud()
            else:
                self.output_dir = cfg.output_uri
                make_dir(self.output_dir)

            self.last_model_path = join(self.output_dir, 'last-model.pth')
            self.config_path = join(self.output_dir, 'config.json')
            self.train_state_path = join(self.output_dir, 'train-state.json')
            self.log_path = join(self.output_dir, 'log.csv')
            self.model_bundle_path = join(self.output_dir, 'model-bundle.zip')
            self.metric_names = self.build_metric_names()

            json_to_file(self.cfg.dict(), self.config_path)
            self.load_init_weights()
            self.load_checkpoint()
            self.opt = self.build_optimizer()
            self.build_data()
            self.start_epoch = self.get_start_epoch()
            self.steps_per_epoch = len(
                self.train_ds) // self.cfg.solver.batch_sz
            self.step_scheduler = self.build_step_scheduler()
            self.epoch_scheduler = self.build_epoch_scheduler()

    def main(self):
        cfg = self.cfg
        self.log_data_stats()
        if not cfg.predict_mode:
            self.plot_dataloaders()
            if cfg.overfit_mode:
                self.overfit()
            else:
                self.train()
                if cfg.save_model_bundle:
                    self.save_model_bundle()

        self.load_checkpoint()
        if cfg.eval_train:
            self.eval_model('train')
        self.eval_model('test')
        self.sync_to_cloud()

    def sync_to_cloud(self):
        if self.cfg.output_uri.startswith('s3://'):
            sync_to_dir(self.output_dir, self.cfg.output_uri)

    def sync_from_cloud(self):
        if self.cfg.output_uri.startswith('s3://'):
            sync_from_dir(self.cfg.output_uri, self.output_dir)

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def build_data(self):
        pass

    def log_data_stats(self):
        if self.train_ds:
            print('train_ds: {} items'.format(len(self.train_ds)))
        if self.valid_ds:
            print('valid_ds: {} items'.format(len(self.valid_ds)))
        if self.test_ds:
            print('test_ds: {} items'.format(len(self.test_ds)))

    def build_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.cfg.solver.lr)

    def build_step_scheduler(self):
        scheduler = None
        cfg = self.cfg
        if cfg.solver.one_cycle and cfg.solver.num_epochs > 1:
            total_steps = cfg.solver.num_epochs * self.steps_per_epoch
            step_sz_up = (cfg.solver.num_epochs // 2) * self.steps_per_epoch
            step_sz_down = total_steps - step_sz_up
            step_scheduler = CyclicLR(
                self.opt,
                base_lr=cfg.solver.lr / 10,
                max_lr=cfg.solver.lr,
                step_sz_up=step_sz_up,
                step_sz_down=step_sz_down,
                cycle_momentum=False)
            for _ in range(self.start_epoch * self.steps_per_epoch):
                step_scheduler.step()
        return scheduler

    def build_epoch_scheduler(self):
        scheduler = None
        if self.cfg.solver.multi_stage:
            scheduler = MultiStepLR(
                self.opt, milestones=self.cfg.solver.multi_stage, gamma=0.1)
            for _ in range(self.start_epoch):
                scheduler.step()
        return scheduler

    def build_metric_names(self):
        metric_names = [
            'epoch', 'train_time', 'valid_time', 'train_loss', 'val_loss',
            'avg_f1', 'avg_precision', 'avg_recall'
        ]

        for label in self.cfg.data.labels:
            metric_names.extend([
                '{}_f1'.format(label), '{}_precision'.format(label),
                '{}_recall'.format(label)
            ])
        return metric_names

    @abstractmethod
    def train_step(self, x, y):
        pass

    @abstractmethod
    def validate_step(self, x, y):
        pass

    def train_end(self, outputs, num_samples):
        metrics = {}
        for k in outputs[0].keys():
            metrics[k] = torch.stack([o[k] for o in outputs
                                      ]).sum().item() / num_samples
        return metrics

    def validate_end(self, outputs, num_samples):
        metrics = {}
        for k in outputs[0].keys():
            metrics[k] = torch.stack([o[k] for o in outputs
                                      ]).sum().item() / num_samples
        return metrics

    def post_forward(self, x):
        return x

    def to_batch(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x

    def normalize_input(self, x):
        return x.float() / 255.0

    def predict(self, x, normalize=False):
        x = self.to_batch(x)
        if normalize:
            x = self.normalize_input(x)
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
            out = self.post_forward(out)
        out = out.cpu()
        return out

    def numpy_predict(self, x):
        """Make a prediction using a TF-formatted (ie. channels last) numpy array.

        Args:
            x: (ndarray) of shape [height, width, channels] or
                [batch_sz, height, width, channels]

        Returns:
            ndarray of shape [batch_sz, num_features]
        """
        x = torch.tensor(x)
        x = self.to_batch(x)
        x = x.permute((0, 3, 1, 2))
        out = self.predict(x, normalize=True)
        return out.numpy()

    def predict_dataloader(self, dl, one_batch=False, return_x=True):
        self.model.eval()

        xs, ys, zs = [], [], []
        with torch.no_grad():
            for x, y in dl:
                x = x.to(self.device)
                z = self.post_forward(self.model(x))
                x = x.cpu()
                z = z.cpu()
                if one_batch:
                    return x, y, z
                if return_x:
                    xs.append(x)
                ys.append(y)
                zs.append(z)

        if return_x:
            return torch.cat(xs), torch.cat(ys), torch.cat(zs)
        return torch.cat(ys), torch.cat(zs)

    def get_dataloader(self, split):
        if split == 'train':
            return self.train_dl
        elif split == 'valid':
            return self.valid_dl
        elif split == 'test':
            return self.test_dl
        else:
            raise ValueError('{} is not a valid split'.format(split))

    @abstractmethod
    def plot_xyz(self, ax, x, y, z=None):
        pass

    def plot_batch(self, x, y, output_path, z=None):
        batch_sz = x.shape[0]
        ncols = nrows = math.ceil(math.sqrt(batch_sz))
        fig = plt.figure(
            constrained_layout=True, figsize=(3 * ncols, 3 * nrows))
        grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

        for i in range(batch_sz):
            ax = fig.add_subplot(grid[i])
            if z is None:
                self.plot_xyz(ax, x[i], y[i])
            else:
                self.plot_xyz(ax, x[i], y[i], z=z[i])

        make_dir(output_path, use_dirname=True)
        plt.savefig(output_path)
        plt.close()

    def plot_predictions(self, split):
        print('Plotting predictions...')
        dl = self.get_dataloader(split)
        output_path = join(self.output_dir, '{}_preds.png'.format(split))
        x, y, z = self.predict_dataloader(dl, one_batch=True)
        self.plot_batch(x, y, output_path, z=z)

    def plot_dataloader(self, dl, output_path):
        x, y = next(iter(dl))
        self.plot_batch(x, y, output_path)

    def plot_dataloaders(self):
        if self.train_dl:
            self.plot_dataloader(
                self.train_dl, join(self.output_dir, 'dataloaders/train.png'))
        if self.valid_dl:
            self.plot_dataloader(
                self.valid_dl, join(self.output_dir, 'dataloaders/valid.png'))
        if self.test_dl:
            self.plot_dataloader(self.test_dl,
                                 join(self.output_dir, 'dataloaders/test.png'))

    @staticmethod
    def from_model_bundle(model_bundle_uri, tmp_dir):
        model_bundle_path = download_if_needed(model_bundle_uri, tmp_dir)
        model_bundle_dir = join(tmp_dir, 'model-bundle')
        unzip(model_bundle_path, model_bundle_dir)

        config_path = join(model_bundle_dir, 'config.json')
        model_path = join(model_bundle_dir, 'model.pth')
        cfg = build_config(file_to_json(config_path))
        return cfg.get_learner()(cfg, tmp_dir, model_path=model_path)

    def save_model_bundle(self):
        model_bundle_dir = join(self.tmp_dir, 'model-bundle')
        make_dir(model_bundle_dir)
        shutil.copyfile(self.last_model_path,
                        join(model_bundle_dir, 'model.pth'))
        shutil.copyfile(self.config_path, join(model_bundle_dir,
                                               'config.json'))
        zipdir(model_bundle_dir, self.model_bundle_path)

    def get_start_epoch(self):
        start_epoch = 0
        if isfile(self.log_path):
            with open(self.log_path) as log_file:
                last_line = log_file.readlines()[-1]
            last_epoch = int(last_line.split(',')[0].strip())
            start_epoch = last_epoch + 1
        return start_epoch

    def load_init_weights(self):
        if self.cfg.model.init_weights:
            weights_path = download_if_needed(self.cfg.model.init_weights,
                                              self.tmp_dir)
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device))

    def load_checkpoint(self):
        if isfile(self.last_model_path):
            print('Loading checkpoint from {}'.format(self.last_model_path))
            self.model.load_state_dict(
                torch.load(self.last_model_path, map_location=self.device))

    def train_epoch(self):
        start = time.time()
        self.model.train()
        num_samples = 0
        outputs = []
        with click.progressbar(self.train_dl, label='Training') as bar:
            for batch_ind, (x, y) in enumerate(bar):
                x = x.to(self.device)
                y = y.to(self.device)
                batch = (x, y)
                self.opt.zero_grad()
                output = self.train_step(batch, batch_ind)
                outputs.append(output)
                loss = output['train_loss']
                loss.backward()
                self.opt.step()
                if self.step_scheduler:
                    self.step_scheduler.step()
                num_samples += x.shape[0]
        metrics = self.train_end(outputs, num_samples)
        end = time.time()
        train_time = datetime.timedelta(seconds=end - start)
        metrics['train_time'] = str(train_time)
        return metrics

    def validate_epoch(self, dl):
        start = time.time()
        self.model.eval()
        num_samples = 0
        outputs = []
        with torch.no_grad():
            with click.progressbar(dl, label='Validating') as bar:
                for batch_ind, (x, y) in enumerate(bar):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    batch = (x, y)
                    output = self.validate_step(batch, batch_ind)
                    outputs.append(output)
                    num_samples += x.shape[0]
        end = time.time()
        validate_time = datetime.timedelta(seconds=end - start)

        metrics = self.validate_end(outputs, num_samples)
        metrics['valid_time'] = str(validate_time)
        return metrics

    def overfit(self):
        self.on_overfit_start()

        x, y = next(iter(self.train_dl))
        x = x.to(self.device)
        y = y.to(self.device)
        batch = (x, y)

        with click.progressbar(
                range(self.cfg.solver.overfit_num_steps),
                label='Overfitting') as bar:
            for step in bar:
                loss = self.train_step(batch, step)['train_loss']
                loss.backward()
                self.opt.step()

                if (step + 1) % 25 == 0:
                    print('\nstep: {}'.format(step))
                    print('train_loss: {}'.format(loss))

        torch.save(self.model.state_dict(), self.last_model_path)

    def train(self):
        self.on_train_start()

        if self.start_epoch > 0 and self.start_epoch <= self.cfg.solver.num_epochs:
            print('Resuming training from epoch {}'.format(self.start_epoch))

        for epoch in range(self.start_epoch, self.cfg.solver.num_epochs):
            print('----------------------------------------')
            print('epoch: {}'.format(epoch), flush=True)
            train_metrics = self.train_epoch()
            if self.epoch_scheduler:
                self.epoch_scheduler.step()
            valid_metrics = self.validate_epoch(self.valid_dl)
            metrics = dict(epoch=epoch, **train_metrics, **valid_metrics)
            print('metrics: {}'.format(metrics), flush=True)
            print()

            self.on_epoch_end(epoch, metrics)

    def on_overfit_start(self):
        pass

    def on_train_start(self):
        pass

    def on_epoch_end(self, curr_epoch, metrics):
        if not isfile(self.log_path):
            with open(self.log_path, 'w') as log_file:
                log_writer = csv.writer(log_file)
                row = self.metric_names
                log_writer.writerow(row)

        with open(self.log_path, 'a') as log_file:
            log_writer = csv.writer(log_file)
            row = [metrics[k] for k in self.metric_names]
            log_writer.writerow(row)

        torch.save(self.model.state_dict(), self.last_model_path)

        if (curr_epoch + 1) % self.cfg.solver.sync_interval == 0:
            self.sync_to_cloud()

    def eval_model(self, split):
        print('Evaluating on {} set...'.format(split))
        dl = self.get_dataloader(split)
        metrics = self.validate_epoch(dl)
        print('metrics: {}'.format(metrics))
        json_to_file(metrics,
                     join(self.output_dir, '{}_metrics.json'.format(split)))
        self.plot_predictions(split)
