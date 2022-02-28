from typing import Iterable, Optional
import warnings
warnings.filterwarnings('ignore')  # noqa

import logging

import matplotlib
matplotlib.use('Agg')  # noqa
from albumentations import BboxParams

import numpy as np
import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils.utils import adjust_conv_channels
from rastervision.pytorch_learner.object_detection_utils import (
    BoxList, TorchVisionODAdapter, compute_coco_eval, collate_fn, plot_xyz)

log = logging.getLogger(__name__)


class ObjectDetectionLearner(Learner):
    def build_model(self) -> FasterRCNN:
        """Returns a FasterRCNN model.

        Note that the model returned will have (num_classes + 2) output classes.
        +1 for the null class (zeroth index), and another +1 (last index) for
        backward compatibility with earlier Raster Vision versions.

        Returns:
            FasterRCNN: a FasterRCNN model.
        """
        pretrained = self.cfg.model.pretrained
        backbone_arch = self.cfg.model.get_backbone_str()
        img_sz = self.cfg.data.img_sz
        num_classes = len(self.cfg.data.class_names)
        in_channels = self.cfg.data.img_channels

        backbone = resnet_fpn_backbone(backbone_arch, pretrained)

        # default values from FasterRCNN constructor
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        if in_channels != 3:
            extra_channels = in_channels - backbone.body['conv1'].in_channels

            # adjust channels
            backbone.body['conv1'] = adjust_conv_channels(
                old_conv=backbone.body['conv1'],
                in_channels=in_channels,
                pretrained=pretrained)

            # adjust stats
            if extra_channels < 0:
                image_mean = image_mean[:extra_channels]
                image_std = image_std[:extra_channels]
            else:
                # arbitrarily set mean and stds of the new channels to
                # something similar to the values of the other 3 channels
                image_mean = image_mean + [.45] * extra_channels
                image_std = image_std + [.225] * extra_channels

        model = FasterRCNN(
            backbone=backbone,
            # +1 because torchvision detection models reserve 0 for the null
            # class, another +1 for backward compatibility with earlier Raster
            # Vision versions
            num_classes=num_classes + 1 + 1,
            # TODO we shouldn't need to pass the image size here
            min_size=img_sz,
            max_size=img_sz,
            image_mean=image_mean,
            image_std=image_std)
        return model

    def setup_model(self, model_def_path: Optional[str] = None) -> None:
        """Override to apply the TorchVisionODAdapter wrapper."""
        ext_cfg = self.cfg.model.external_def
        if ext_cfg is not None:
            model = self.load_external_model(ext_cfg, model_def_path)
            # this model will have 1 extra output classes that we will ignore
            self.model = TorchVisionODAdapter(model, ignored_output_inds=[0])
        else:
            model = self.build_model()
            # this model will have 2 extra output classes that we will ignore
            num_classes = len(self.cfg.data.class_names)
            self.model = TorchVisionODAdapter(
                model, ignored_output_inds=[0, num_classes + 1])

        self.model.to(self.device)
        self.load_init_weights()

    def build_metric_names(self):
        metric_names = [
            'epoch', 'train_time', 'valid_time', 'train_loss', 'map', 'map50'
        ]
        return metric_names

    def get_bbox_params(self):
        return BboxParams(
            format='albumentations', label_fields=['category_id'])

    def get_collate_fn(self):
        return collate_fn

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

    def numpy_predict(self, x: np.ndarray,
                      raw_out: bool = False) -> np.ndarray:
        transform, _ = self.get_data_transforms()
        x = self.normalize_input(x)
        x = self.to_batch(x)
        x = np.stack([
            transform(image=img, bboxes=[], category_id=[])['image']
            for img in x
        ])
        x = torch.from_numpy(x)
        x = x.permute((0, 3, 1, 2))
        out = self.predict(x, raw_out=raw_out)
        return self.output_to_numpy(out)

    def output_to_numpy(self, out: Iterable[BoxList]):
        numpy_out = []
        for boxlist in out:
            numpy_out.append({
                'boxes':
                boxlist.convert_boxes('yxyx').numpy(),
                'class_ids':
                boxlist.get_field('class_ids').numpy(),
                'scores':
                boxlist.get_field('scores').numpy()
            })
        return numpy_out

    def plot_xyz(self, ax, x, y, z=None):
        data_cfg = self.cfg.data
        plot_xyz(
            ax,
            x,
            y,
            class_names=data_cfg.class_names,
            class_colors=data_cfg.class_colors,
            z=z)

    def prob_to_pred(self, x):
        return x
