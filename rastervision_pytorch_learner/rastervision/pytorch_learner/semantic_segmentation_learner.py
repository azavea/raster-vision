from typing import TYPE_CHECKING, Optional, Tuple
import warnings

import logging

import torch
from torch.nn import functional as F
import torch.distributed as dist

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.utils import (
    compute_conf_mat_metrics, compute_conf_mat, aggregate_metrics)
from rastervision.pytorch_learner.dataset.visualizer import (
    SemanticSegmentationVisualizer)

if TYPE_CHECKING:
    from torch import nn

warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)


class SemanticSegmentationLearner(Learner):
    def get_visualizer_class(self):
        return SemanticSegmentationVisualizer

    def train_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        return {'train_loss': self.loss(out, y)}

    def validate_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        val_loss = self.loss(out, y)

        num_labels = len(self.cfg.data.class_names)
        y = y.view(-1)
        out = self.prob_to_pred(out).view(-1)
        conf_mat = compute_conf_mat(out, y, num_labels)

        return {'val_loss': val_loss, 'conf_mat': conf_mat}

    def validate_end(self, outputs):
        metrics = aggregate_metrics(outputs, exclude_keys={'conf_mat'})
        conf_mat = sum([o['conf_mat'] for o in outputs])

        if self.is_ddp_process:
            metrics = self.reduce_distributed_metrics(metrics)
            dist.reduce(conf_mat, dst=0, op=dist.ReduceOp.SUM)
            if not self.is_ddp_master:
                return metrics

        ignored_idx = self.cfg.solver.ignore_class_index
        if ignored_idx is not None and ignored_idx < 0:
            ignored_idx += self.cfg.data.num_classes

        class_names = self.cfg.data.class_names
        conf_mat_metrics = compute_conf_mat_metrics(
            conf_mat, class_names, ignore_idx=ignored_idx)

        metrics.update(conf_mat_metrics)
        return metrics

    def post_forward(self, x):
        if isinstance(x, dict):
            return x['out']
        return x

    def predict(self,
                x: torch.Tensor,
                raw_out: bool = False,
                out_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if out_shape is None:
            out_shape = x.shape[-2:]

        x = self.to_batch(x).float()
        x = self.to_device(x, self.device)

        with torch.inference_mode():
            out = self.model(x)
            out = self.post_forward(out)
        out = self.postprocess_model_output(
            out, raw_out=raw_out, out_shape=out_shape)
        return out

    def predict_onnx(
            self,
            x: torch.Tensor,
            raw_out: bool = False,
            out_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:

        if out_shape is None:
            out_shape = x.shape[-2:]

        x = self.to_batch(x).float()
        out = self.model(x)
        out = self.post_forward(out)
        out = self.postprocess_model_output(
            out, raw_out=raw_out, out_shape=out_shape)
        return out

    def postprocess_model_output(self, out: torch.Tensor, raw_out: bool,
                                 out_shape: Tuple[int, int]):
        out = out.softmax(dim=1)
        # ensure correct output shape
        if out.shape[-2:] != out_shape:
            out = F.interpolate(
                out, size=out_shape, mode='bilinear', align_corners=False)

        if not raw_out:
            out = self.prob_to_pred(out)
        out = self.to_device(out, 'cpu')

        return out

    def prob_to_pred(self, x):
        return x.argmax(1)

    def export_to_onnx(self,
                       path: str,
                       model: Optional['nn.Module'] = None,
                       sample_input: Optional[torch.Tensor] = None,
                       **kwargs) -> None:
        args = dict(
            input_names=['x'],
            output_names=['out'],
            dynamic_axes={
                'x': {
                    0: 'batch_size',
                    2: 'height',
                    3: 'width',
                },
                'out': {
                    0: 'batch_size',
                    2: 'height',
                    3: 'width',
                },
            },
        )
        args.update(kwargs)
        return super().export_to_onnx(path, model, sample_input, **args)
