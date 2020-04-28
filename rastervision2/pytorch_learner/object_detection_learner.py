import warnings
warnings.filterwarnings('ignore')  # noqa
from os.path import join, isdir
import logging

import matplotlib
matplotlib.use('Agg')  # noqa
from torch.utils.data import ConcatDataset
from albumentations import BboxParams

from rastervision2.pytorch_learner.learner import Learner
from rastervision2.pytorch_learner.object_detection_utils import (
    MyFasterRCNN, CocoDataset, compute_coco_eval, collate_fn, plot_xyz)
from rastervision2.pytorch_learner.object_detection_learner_config import (
    ObjectDetectionDataFormat)

log = logging.getLogger(__name__)


class ObjectDetectionLearner(Learner):
    def build_model(self):
        # TODO we shouldn't need to pass the image size here
        pretrained = self.cfg.model.pretrained
        model = MyFasterRCNN(
            self.cfg.model.get_backbone_str(),
            len(self.cfg.data.class_names),
            self.cfg.data.img_sz,
            pretrained=pretrained)
        return model

    def build_metric_names(self):
        metric_names = [
            'epoch', 'train_time', 'valid_time', 'train_loss', 'map', 'map50'
        ]
        return metric_names

    def get_bbox_params(self):
        return BboxParams(format='coco', label_fields=['category_id'])

    def get_collate_fn(self):
        return collate_fn

    def get_datasets(self):
        cfg = self.cfg

        if cfg.data.data_format == ObjectDetectionDataFormat.default:
            data_dirs = self.unzip_data()

        transform, aug_transform = self.get_data_transforms()

        train_ds, valid_ds, test_ds = [], [], []
        for data_dir in data_dirs:
            train_dir = join(data_dir, 'train')
            valid_dir = join(data_dir, 'valid')

            if isdir(train_dir):
                img_dir = join(train_dir, 'img')
                annotation_uri = join(train_dir, 'labels.json')
                if cfg.overfit_mode:
                    train_ds.append(
                        CocoDataset(
                            img_dir, annotation_uri, transform=transform))
                else:
                    train_ds.append(
                        CocoDataset(
                            img_dir, annotation_uri, transform=aug_transform))

            if isdir(valid_dir):
                img_dir = join(valid_dir, 'img')
                annotation_uri = join(valid_dir, 'labels.json')
                valid_ds.append(
                    CocoDataset(img_dir, annotation_uri, transform=transform))
                test_ds.append(
                    CocoDataset(img_dir, annotation_uri, transform=transform))

        train_ds, valid_ds, test_ds = \
            ConcatDataset(train_ds), ConcatDataset(valid_ds), ConcatDataset(test_ds)

        return train_ds, valid_ds, test_ds

    def train_step(self, batch, batch_ind):
        x, y = batch
        loss_dict = self.model(x, y)
        return {'train_loss': loss_dict['total_loss']}

    def validate_step(self, batch, batch_ind):
        x, y = batch
        outs = self.model(x)
        ys = self.to_device(y, 'cpu')
        outs = self.to_device(outs, 'cpu')

        return {'ys': ys, 'outs': outs}

    def validate_end(self, outputs, num_samples):
        outs = []
        ys = []
        for o in outputs:
            outs.extend(o['outs'])
            ys.extend(o['ys'])
        num_class_ids = len(self.cfg.data.class_names)
        coco_eval = compute_coco_eval(outs, ys, num_class_ids)

        metrics = {'map': 0.0, 'map50': 0.0}
        if coco_eval is not None:
            coco_metrics = coco_eval.stats
            metrics = {'map': coco_metrics[0], 'map50': coco_metrics[1]}
        return metrics

    def output_to_numpy(self, out):
        numpy_out = []
        for boxlist in out:
            boxlist = boxlist.cpu()
            numpy_out.append({
                'boxes':
                boxlist.boxes.numpy(),
                'class_ids':
                boxlist.get_field('class_ids').numpy(),
                'scores':
                boxlist.get_field('scores').numpy()
            })
        return numpy_out

    def plot_xyz(self, ax, x, y, z=None):
        plot_xyz(ax, x, y, self.cfg.data.class_names, z=z)

    def prob_to_pred(self, x):
        return x
