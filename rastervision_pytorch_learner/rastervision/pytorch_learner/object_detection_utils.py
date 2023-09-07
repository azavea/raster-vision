from typing import (Any, Callable, Optional, Sequence, Tuple, Iterable, List,
                    Dict, Union)
from collections import defaultdict
from os.path import join
from operator import iand
from functools import reduce
from pprint import pformat

import torch
import torch.nn as nn
from torchvision.ops import (box_area, box_convert, batched_nms,
                             clip_boxes_to_image)
from torchvision.utils import draw_bounding_boxes
import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

from rastervision.pipeline.file_system import json_to_file, get_tmp_dir
from rastervision.pytorch_learner.utils.utils import ONNXRuntimeAdapter


def get_coco_gt(targets: Iterable['BoxList'],
                num_class_ids: int) -> Dict[str, List[dict]]:
    images = []
    annotations = []
    ann_id = 1
    for img_id, target in enumerate(targets, 1):
        # Use fake height, width, and filename because they don't matter.
        images.append({
            'id': img_id,
            'height': 1000,
            'width': 1000,
            'file_name': '{}.png'.format(img_id)
        })
        boxes = target.convert_boxes('xywh').float().tolist()
        class_ids = target.get_field('class_ids').tolist()
        areas = box_area(target.boxes).tolist()
        for box, class_id, area in zip(boxes, class_ids, areas):
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'bbox': box,
                'category_id': class_id + 1,
                'area': area,
                'iscrowd': 0
            })
            ann_id += 1

    categories = [{
        'id': class_id + 1,
        'name': str(class_id + 1),
        'supercategory': 'super'
    } for class_id in range(num_class_ids)]
    coco = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    return coco


def get_coco_preds(outputs: Iterable['BoxList']) -> List[dict]:
    preds = []
    for img_id, output in enumerate(outputs, 1):
        boxes = output.convert_boxes('xywh').float().tolist()
        class_ids = output.get_field('class_ids').tolist()
        scores = output.get_field('scores').tolist()
        for box, class_id, score in zip(boxes, class_ids, scores):
            preds.append({
                'image_id': img_id,
                'category_id': class_id + 1,
                'bbox': box,
                'score': score
            })
    return preds


def compute_coco_eval(outputs, targets, num_class_ids):
    """Return mAP averaged over 0.5-0.95 using pycocotools eval.

    Note: boxes are in (ymin, xmin, ymax, xmax) format with values ranging
        from 0 to h or w.

    Args:
        outputs: (list) of length m containing dicts of form
            {'boxes': <tensor with shape (n, 4)>,
             'class_ids': <tensor with shape (n,)>,
             'scores': <tensor with shape (n,)>}
        targets: (list) of length m containing dicts of form
            {'boxes': <tensor with shape (n, 4)>,
             'class_ids': <tensor with shape (n,)>}
    """
    with get_tmp_dir() as tmp_dir:
        preds = get_coco_preds(outputs)
        # ap is undefined when there are no predicted boxes
        if len(preds) == 0:
            return None

        gt = get_coco_gt(targets, num_class_ids)
        gt_path = join(tmp_dir, 'gt.json')
        json_to_file(gt, gt_path)
        coco_gt = COCO(gt_path)

        pycocotools.coco.unicode = None
        coco_preds = coco_gt.loadRes(preds)

        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_preds, iouType='bbox')

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval


class BoxList():
    def __init__(self, boxes: torch.Tensor, format: str = 'xyxy',
                 **extras) -> None:
        """Representation of a list of bounding boxes and associated data.

        Internally, boxes are always stored in the xyxy format.

        Args:
            boxes: tensor<n, 4>
            format: format of input boxes.
            extras: dict with values that are tensors with first dimension corresponding
                to boxes first dimension
        """
        self.extras = extras
        if format == 'xyxy':
            self.boxes = boxes
        elif format == 'yxyx':
            self.boxes = boxes[:, [1, 0, 3, 2]]
        else:
            self.boxes = box_convert(boxes, format, 'xyxy')

    def __contains__(self, key: str) -> bool:
        return key == 'boxes' or key in self.extras

    def get_field(self, name: str) -> Any:
        if name == 'boxes':
            return self.boxes
        else:
            return self.extras.get(name)

    def _map_extras(self, func: Callable,
                    cond: Callable = lambda k, v: True) -> dict:
        new_extras = {}
        for k, v in self.extras.items():
            if cond(k, v):
                new_extras[k] = func(k, v)
            else:
                new_extras[k] = v

        return new_extras

    def copy(self) -> 'BoxList':
        return BoxList(
            self.boxes.copy(),
            **self._map_extras(lambda k, v: v.copy()),
            cond=lambda k, v: torch.is_tensor(v))

    def to(self, *args, **kwargs) -> 'BoxList':
        """Recursively apply :meth:`torch.Tensor.to` to Tensors.

        Args:
            *args: Args for :meth:`torch.Tensor.to`.
            **kwargs: Keyword args for :meth:`torch.Tensor.to`.

        Returns:
            BoxList: New BoxList with to'd Tensors.
        """
        boxes = self.boxes.to(*args, **kwargs)
        extras = self._map_extras(
            func=lambda k, v: v.to(*args, **kwargs),
            cond=lambda k, v: torch.is_tensor(v))
        return BoxList(boxes, **extras)

    def convert_boxes(self, out_fmt: str) -> torch.Tensor:
        if out_fmt == 'yxyx':
            boxes = self.boxes[:, [1, 0, 3, 2]]
        else:
            boxes = box_convert(self.boxes, 'xyxy', out_fmt)
        return boxes

    def __len__(self) -> int:
        return len(self.boxes)

    @staticmethod
    def cat(box_lists: Iterable['BoxList']) -> 'BoxList':
        boxes = []
        extras = defaultdict(list)
        for bl in box_lists:
            boxes.append(bl.boxes)
            for k, v in bl.extras.items():
                extras[k].append(v)
        boxes = torch.cat(boxes)
        for k, v in extras.items():
            extras[k] = torch.cat(v)
        return BoxList(boxes, **extras)

    def equal(self, other: 'BoxList') -> bool:
        if len(other) != len(self):
            return False

        # Ignore order of boxes.
        extras = [(v.float().unsqueeze(1) if v.ndim == 1 else v.float())
                  for v in self.extras.values()]
        cat_arr = torch.cat([self.boxes] + extras, 1)
        self_tups = set([tuple([x.item() for x in row]) for row in cat_arr])

        extras = [(v.float().unsqueeze(1) if v.ndim == 1 else v.float())
                  for v in other.extras.values()]
        cat_arr = torch.cat([other.boxes] + extras, 1)
        other_tups = set([tuple([x.item() for x in row]) for row in cat_arr])
        return self_tups == other_tups

    def ind_filter(self, inds: Sequence[int]) -> 'BoxList':
        boxes = self.boxes[inds]
        extras = self._map_extras(
            func=lambda k, v: v[inds], cond=lambda k, v: torch.is_tensor(v))
        return BoxList(boxes, **extras)

    def score_filter(self, score_thresh: float = 0.25) -> 'BoxList':
        scores = self.extras.get('scores')
        if scores is not None:
            return self.ind_filter(scores > score_thresh)
        else:
            raise ValueError('must have scores as key in extras')

    def clip_boxes(self, img_height: int, img_width: int) -> 'BoxList':
        boxes = clip_boxes_to_image(self.boxes, (img_height, img_width))
        return BoxList(boxes, **self.extras)

    def nms(self, iou_thresh: float = 0.5) -> torch.Tensor:
        if len(self) == 0:
            return self

        good_inds = batched_nms(self.boxes, self.get_field('scores'),
                                self.get_field('class_ids'), iou_thresh)
        return self.ind_filter(good_inds)

    def scale(self, yscale: float, xscale: float) -> 'BoxList':
        """Scale box coords by the given scaling factors."""
        dtype = self.boxes.dtype
        boxes = self.boxes.float()
        boxes[:, [0, 2]] *= xscale
        boxes[:, [1, 3]] *= yscale
        self.boxes = boxes.to(dtype=dtype)
        return self

    def pin_memory(self) -> 'BoxList':
        self.boxes = self.boxes.pin_memory()
        for k, v in self.extras.items():
            if torch.is_tensor(v):
                self.extras[k] = v.pin_memory()
        return self

    def __repr__(self) -> str:  # pragma: no cover
        return pformat(dict(boxes=self.boxes, **self.extras))


def collate_fn(data: Iterable[Sequence]) -> Tuple[torch.Tensor, List[BoxList]]:
    imgs = [d[0] for d in data]
    x = torch.stack(imgs)
    y: List[BoxList] = [d[1] for d in data]
    return x, y


def draw_boxes(x: torch.Tensor, y: BoxList, class_names: Sequence[str],
               class_colors: Sequence[str]) -> torch.Tensor:
    """Given an image and a BoxList, draw the boxes in the BoxList on the
    image."""
    boxes = y.boxes
    class_ids: np.ndarray = y.get_field('class_ids').numpy()
    scores: Optional[torch.Tensor] = y.get_field('scores')

    if len(boxes) > 0:
        box_annotations: List[str] = np.array(class_names)[class_ids].tolist()
        if scores is not None:
            box_annotations = [
                f'{ann} | {score:.2f}'
                for ann, score in zip(box_annotations, scores)
            ]
        box_colors: List[Union[str, Tuple[int, ...]]] = [
            tuple(c) if not isinstance(c, str) else c
            for c in np.array(class_colors)[class_ids]
        ]

        # convert image to uint8
        if x.is_floating_point():
            x = (x * 255).byte()
        x = x.permute(2, 0, 1)
        x = draw_bounding_boxes(
            image=x,
            boxes=boxes,
            labels=box_annotations,
            colors=box_colors,
            width=2)
        x = x.permute(1, 2, 0) / 255.

    return x


class TorchVisionODAdapter(nn.Module):
    """Adapter for interfacing with TorchVision's object detection models.

    The purpose of this adapter is:
    1) to convert input BoxLists to dicts before feeding them into the model
    2) to convert detections output by the model as dicts into BoxLists

    Additionally, it automatically converts to/from 1-indexed class labels
    (which is what the TorchVision models expect).
    """

    def __init__(self,
                 model: nn.Module,
                 ignored_output_inds: Sequence[int] = [0]) -> None:
        """Constructor.

        Args:
            model (nn.Module): A torchvision object detection model.
            ignored_output_inds (Iterable[int], optional): Class labels to exclude.
                Defaults to [0].
        """
        super().__init__()
        self.model = model
        self.ignored_output_inds = ignored_output_inds

    def forward(self,
                input: torch.Tensor,
                targets: Optional[Iterable[BoxList]] = None
                ) -> Union[Dict[str, Any], List[BoxList]]:
        """Forward pass.

        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): batch
                of images.
            targets (Optional[Iterable[BoxList]], optional): In training mode,
                should be Iterable[BoxList]], with each BoxList having a
                'class_ids' field. In eval mode, should be None. Defaults to
                None.

        Returns:
            In training mode, returns a dict of losses. In eval mode, returns a
            list of BoxLists containing predicted boxes, class_ids, and scores.
            Further filtering based on score should be done before considering
            the prediction "final".
        """
        if targets is not None:
            # Convert each boxlist into the format expected by the torchvision
            # models: a dict with keys, 'boxes' and 'labels'.
            # Note: labels (class IDs) must start at 1.
            _targets = [self.boxlist_to_model_input_dict(bl) for bl in targets]
            loss_dict = self.model(input, _targets)
            return loss_dict

        outs = self.model(input)
        boxlists = [self.model_output_dict_to_boxlist(out) for out in outs]

        return boxlists

    def boxlist_to_model_input_dict(self, boxlist: BoxList) -> dict:
        """Convert BoxList to dict compatible w/ torchvision detection models.

        Also, make class labels 1-indexed.

        Args:
            boxlist (BoxList): A BoxList with a "class_ids" field.

        Returns:
            dict: A dict with keys: "boxes" and "labels".
        """
        return {
            'boxes': boxlist.boxes,
            # make class IDs 1-indexed
            'labels': (boxlist.get_field('class_ids') + 1)
        }

    def model_output_dict_to_boxlist(self, out: dict) -> BoxList:
        """Convert model output dict to BoxList.

        Also, exclude any null classes and make class labels 0-indexed.

        Args:
            out (dict): A dict output by a torchvision detection model in eval
                mode.

        Returns:
            BoxList: A BoxList with "class_ids" and "scores" fields.
        """
        # keep only the detections of the non-null classes
        exclude_masks = [out['labels'] != i for i in self.ignored_output_inds]
        mask = reduce(iand, exclude_masks)
        boxlist = BoxList(
            boxes=out['boxes'][mask],
            # make class IDs 0-indexed again
            class_ids=(out['labels'][mask] - 1),
            scores=out['scores'][mask])
        return boxlist


class ONNXRuntimeAdapterForFasterRCNN(ONNXRuntimeAdapter):
    """TorchVision Faster RCNN model exported as ONNX"""

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        N, *_ = x.shape
        x = x.numpy()
        outputs = self.ort_session.run(None, dict(x=x))
        out_dicts = [None] * N
        for i in range(N):
            boxes, labels, scores = outputs[i * 3:i * 3 + 3]
            boxes = torch.from_numpy(boxes)
            labels = torch.from_numpy(labels)
            scores = torch.from_numpy(scores)
            out_dicts[i] = dict(boxes=boxes, labels=labels, scores=scores)
        return out_dicts
