from typing import (TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple,
                    Union)
import warnings

import logging

import numpy as np
import torch
import torch.distributed as dist

from rastervision.pytorch_learner.learner import Learner
from rastervision.pytorch_learner.object_detection_utils import (
    BoxList, TorchVisionODAdapter, compute_coco_eval, collate_fn,
    ONNXRuntimeAdapterForFasterRCNN)
from rastervision.pytorch_learner.dataset.visualizer import (
    ObjectDetectionVisualizer)

if TYPE_CHECKING:
    from torch import nn, Tensor

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
            img_sz=cfg.data.img_sz,
            ddp_rank=self.ddp_local_rank)
        return model

    def setup_model(self,
                    model_weights_path: Optional[str] = None,
                    model_def_path: Optional[str] = None) -> None:
        """Override to apply the TorchVisionODAdapter wrapper."""
        if self.model is not None:
            self.model.to(self.device)
            return

        self._onnx_mode = (model_weights_path is not None
                           and model_weights_path.lower().endswith('.onnx'))
        if self._onnx_mode:
            model = self.load_onnx_model(model_weights_path)
        else:
            model = self.build_model(model_def_path)

        if self.cfg.model.external_def is not None:
            # this model will have 1 extra output classes that we will ignore
            self.model = TorchVisionODAdapter(model, ignored_output_inds=[0])
        else:
            # this model will have 2 extra output classes that we will ignore
            num_classes = self.cfg.data.num_classes
            self.model = TorchVisionODAdapter(
                model, ignored_output_inds=[0, num_classes + 1])

        if not self._onnx_mode:
            self.model.to(self.device)
            self.load_init_weights(model_weights_path)

    def get_collate_fn(self):
        return collate_fn

    def train_step(self, batch, batch_ind):
        x, y = batch
        loss_dict = self.model(x, y)
        loss_dict['train_loss'] = sum(loss_dict.values())
        return loss_dict

    def validate_step(self, batch, batch_ind):
        x, y = batch
        outs = self.model(x)
        ys = self.to_device(y, 'cpu')
        outs = self.to_device(outs, 'cpu')

        return {'ys': ys, 'outs': outs}

    def validate_end(self, outputs):
        outs = []
        ys = []
        for o in outputs:
            outs.extend(o['outs'])
            ys.extend(o['ys'])
        num_class_ids = len(self.cfg.data.class_names)

        if self.is_ddp_process:
            is_master = self.is_ddp_master
            all_outs = [None] * self.ddp_world_size
            all_ys = [None] * self.ddp_world_size
            dist.gather_object(
                outs,
                object_gather_list=(all_outs if is_master else None),
                dst=0)
            dist.gather_object(
                ys, object_gather_list=(all_ys if is_master else None), dst=0)
            if not is_master:
                return {}
            outs = sum(all_outs, [])
            ys = sum(all_ys, [])

        log.info(f'{self.ddp_rank} at coco eval')
        coco_eval = compute_coco_eval(outs, ys, num_class_ids)

        metrics = {'mAP': 0.0, 'mAP50': 0.0}
        if coco_eval is not None:
            coco_metrics = coco_eval.stats
            metrics = {'mAP': coco_metrics[0], 'mAP50': coco_metrics[1]}
        return metrics

    def predict(self,
                x: 'Tensor',
                raw_out: bool = False,
                out_shape: Optional[Tuple[int, int]] = None) -> BoxList:
        """Make prediction for an image or batch of images.

        Args:
            x (Tensor): Image or batch of images as a float Tensor with pixel
                values normalized to [0, 1].
            raw_out (bool, optional): If True, return prediction probabilities.
                Defaults to False.
            out_shape (Optional[Tuple[int, int]], optional): If provided,
                boxes are resized such that they reference pixel coordinates in
                an image of this shape. Defaults to None.

        Returns:
            BoxList: Predicted boxes.
        """
        out: List[BoxList] = super().predict(x, raw_out=raw_out)
        out = self.postprocess_model_output(x, out, out_shape=out_shape)
        return out

    def predict_onnx(self,
                     x: 'Tensor',
                     raw_out: bool = False,
                     out_shape: Optional[Tuple[int, int]] = None) -> BoxList:
        out: List[BoxList] = super().predict(x, raw_out=raw_out)
        out = self.postprocess_model_output(x, out, out_shape=out_shape)
        return out

    def postprocess_model_output(self, x: 'Tensor', out_batch: torch.Tensor,
                                 out_shape: Tuple[int, int]):
        if out_shape is None:
            return out_batch
        h_in, w_in = x.shape[-2:]
        h_out, w_out = out_shape
        yscale, xscale = (h_out / h_in), (w_out / w_in)
        with torch.inference_mode():
            for out in out_batch:
                out.scale(yscale, xscale)
        return out_batch

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

    def export_to_onnx(self,
                       path: str,
                       model: Optional['nn.Module'] = None,
                       sample_input: Optional[torch.Tensor] = None,
                       **kwargs) -> None:
        if model is None and isinstance(self.model, TorchVisionODAdapter):
            model = self.model.model
        return super().export_to_onnx(path, model, sample_input, **kwargs)

    def load_onnx_model(self,
                        model_path: str) -> ONNXRuntimeAdapterForFasterRCNN:
        log.info(f'Loading ONNX model from {model_path}')
        onnx_model = ONNXRuntimeAdapterForFasterRCNN.from_file(model_path)
        return onnx_model
