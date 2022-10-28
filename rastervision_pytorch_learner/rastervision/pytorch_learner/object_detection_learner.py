from typing import (TYPE_CHECKING, Dict, Iterable, List, Optional, Union)
import warnings

import logging

import numpy as np

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.object_detection_utils import (
    BoxList, TorchVisionODAdapter, compute_coco_eval, collate_fn)
from rastervision.pytorch_learner.dataset.visualizer import (
    ObjectDetectionVisualizer)

if TYPE_CHECKING:
    from torch import nn

warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)


class ObjectDetectionLearner(Learner):
    def get_visualizer_class(self):
        return ObjectDetectionVisualizer

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
        self.load_init_weights(model_weights_path)

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

    def output_to_numpy(
            self, out: Iterable[BoxList]
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        def boxlist_to_numpy(boxlist: BoxList) -> Dict[str, np.ndarray]:
            return {
                'boxes': boxlist.convert_boxes('yxyx').numpy(),
                'class_ids': boxlist.get_field('class_ids').numpy(),
                'scores': boxlist.get_field('scores').numpy()
            }

        if isinstance(out, BoxList):
            return boxlist_to_numpy(out)
        else:
            return [boxlist_to_numpy(boxlist) for boxlist in out]

    def prob_to_pred(self, x):
        return x
