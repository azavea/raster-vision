import warnings
warnings.filterwarnings('ignore')  # noqa
from os.path import join

import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

from rastervision.pytorch_learner.learner import Learner


class RegressionModel(nn.Module):
    def __init__(self,
                 backbone_arch,
                 out_features,
                 pretrained=True,
                 pos_out_inds=None,
                 prob_out_inds=None):
        super().__init__()
        self.backbone = getattr(models, backbone_arch)(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, out_features)
        self.pos_out_inds = pos_out_inds
        self.prob_out_inds = prob_out_inds

    def forward(self, x):
        out = self.backbone(x)
        if self.pos_out_inds:
            for ind in self.pos_out_inds:
                out[:, ind] = out[:, ind].exp()
        if self.prob_out_inds:
            for ind in self.prob_out_inds:
                out[:, ind] = out[:, ind].sigmoid()
        return out


class RegressionLearner(Learner):
    def build_model(self):
        pretrained = self.cfg.model.pretrained
        backbone = self.cfg.model.get_backbone_str()
        out_features = len(self.cfg.data.class_names)
        pos_out_inds = [
            self.cfg.data.class_names.index(l)
            for l in self.cfg.data.pos_class_names
        ]
        prob_out_inds = [
            self.cfg.data.class_names.index(l)
            for l in self.cfg.data.prob_class_names
        ]
        model = RegressionModel(
            backbone,
            out_features,
            pretrained=pretrained,
            pos_out_inds=pos_out_inds,
            prob_out_inds=prob_out_inds)
        return model

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
        for label in self.cfg.data.class_names:
            metric_names.extend([
                '{}_abs_error'.format(label),
                '{}_scaled_abs_error'.format(label)
            ])
        return metric_names

    def train_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        return {'train_loss': F.l1_loss(out, y, reduction='sum')}

    def validate_step(self, batch, batch_nb):
        x, y = batch
        out = self.post_forward(self.model(x))
        val_loss = F.l1_loss(out, y, reduction='sum')
        abs_error = torch.abs(out - y).sum(dim=0)
        scaled_abs_error = (
            torch.abs(out - y) / self.target_medians).sum(dim=0)

        metrics = {'val_loss': val_loss}
        for ind, label in enumerate(self.cfg.data.class_names):
            metrics['{}_abs_error'.format(label)] = abs_error[ind]
            metrics['{}_scaled_abs_error'.format(label)] = scaled_abs_error[
                ind]

        return metrics

    def prob_to_pred(self, x):
        return x

    def plot_xyz(self, ax, x, y, z=None):
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

        max_scatter_points = self.cfg.data.plot_options.max_scatter_points
        if y.shape[0] > max_scatter_points:
            scatter_inds = torch.randperm(
                y.shape[0], dtype=torch.long)[0:max_scatter_points]
        else:
            scatter_inds = torch.arange(0, y.shape[0], dtype=torch.long)

        # make scatter plot
        num_labels = len(self.cfg.data.class_names)
        ncols = num_labels
        nrows = 1
        fig = plt.figure(
            constrained_layout=True, figsize=(5 * ncols, 5 * nrows))
        grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

        for label_ind, label in enumerate(self.cfg.data.class_names):
            ax = fig.add_subplot(grid[label_ind])
            ax.scatter(
                y[scatter_inds, label_ind],
                out[scatter_inds, label_ind],
                c='blue',
                alpha=0.1)
            ax.set_title('{} on {} set'.format(label, split))
            ax.set_xlabel('ground truth')
            ax.set_ylabel('predictions')
        scatter_path = join(self.output_dir, '{}_scatter.png'.format(split))
        plt.savefig(scatter_path)
        print('done scatter')

        # make histogram of errors
        fig = plt.figure(
            constrained_layout=True, figsize=(5 * ncols, 5 * nrows))
        grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

        hist_bins = self.cfg.data.plot_options.hist_bins
        for label_ind, label in enumerate(self.cfg.data.class_names):
            ax = fig.add_subplot(grid[label_ind])
            errs = torch.abs(y[:, label_ind] - out[:, label_ind]).tolist()
            ax.hist(errs, bins=hist_bins)
            ax.set_title('{} on {} set'.format(label, split))
            ax.set_xlabel('prediction error')
        hist_path = join(self.output_dir, '{}_err_hist.png'.format(split))
        plt.savefig(hist_path)
        print('done hist')
