from typing import TYPE_CHECKING, Iterable, Optional, Sequence
import warnings

import logging

import numpy as np
import torch

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils.utils import (plot_channel_groups,
                                                      channel_groups_to_imgs)
from rastervision.pytorch_learner.object_detection_utils import (
    BoxList, TorchVisionODAdapter, compute_coco_eval, collate_fn, draw_boxes)

if TYPE_CHECKING:
    from torch import nn

warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)


class ObjectDetectionLearner(Learner):
    def build_model(self, model_def_path: Optional[str] = None) -> 'nn.Module':
        """Override to pass img_sz."""
        cfg = self.cfg
        model = cfg.model.build(
            num_classes=cfg.data.num_classes,
            in_channels=cfg.data.img_channels,
            save_dir=self.modules_dir,
            hubconf_dir=model_def_path,
            img_sz=cfg.data.img_sz)
        return model

    def setup_model(self,
                    model_weights_path: Optional[str] = None,
                    model_def_path: Optional[str] = None) -> None:
        """Override to apply the TorchVisionODAdapter wrapper."""
        if self.model is not None:
            self.model.to(self.device)
            return

        model = self.build_model(model_def_path)

        if self.cfg.model.external_def is not None:
            # this model will have 1 extra output classes that we will ignore
            self.model = TorchVisionODAdapter(model, ignored_output_inds=[0])
        else:
            # this model will have 2 extra output classes that we will ignore
            num_classes = self.cfg.data.num_classes
            self.model = TorchVisionODAdapter(
                model, ignored_output_inds=[0, num_classes + 1])

        self.model.to(self.device)
        self.load_init_weights()

    def build_metric_names(self):
        metric_names = [
            'epoch', 'train_time', 'valid_time', 'train_loss', 'map', 'map50'
        ]
        return metric_names

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
        transform, _ = self.cfg.data.get_data_transforms()
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

    def plot_xyz(self,
                 axs: Sequence,
                 x: torch.Tensor,
                 y: BoxList,
                 z: Optional[BoxList] = None) -> None:

        y = y if z is None else z
        channel_groups = self.cfg.data.plot_options.channel_display_groups

        class_names = self.cfg.data.class_names
        class_colors = self.cfg.data.class_colors

        imgs = channel_groups_to_imgs(x, channel_groups)
        imgs = [draw_boxes(img, y, class_names, class_colors) for img in imgs]
        plot_channel_groups(axs, imgs, channel_groups)

    def prob_to_pred(self, x):
        return x
