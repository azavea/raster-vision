from typing import Optional, Tuple, Any, Callable, Dict
from enum import Enum

import numpy as np
import albumentations as A

import torch

from rastervision.pytorch_learner.object_detection_utils import BoxList


class TransformType(Enum):
    noop = 'noop'
    classification = 'classification'
    regression = 'regression'
    object_detection = 'object_detection'
    semantic_segmentation = 'semantic_segmentation'


def classification_transformer(inp: Tuple[Any, Any],
                               transform=Optional[A.BasicTransform]
                               ) -> Tuple[np.ndarray, np.ndarray]:
    x, y = inp
    x, y = np.array(x), np.array(y)
    if transform is not None:
        out = transform(image=x)
        x = out['image']
    return x, y


def regression_transformer(inp: Tuple[Any, Any],
                           transform=Optional[A.BasicTransform]
                           ) -> Tuple[np.ndarray, np.ndarray]:
    x, y = inp
    x, y = np.array(x), np.array(y)
    if transform is not None:
        out = transform(image=x)
        x = out['image']
    return x, y


def object_detection_transformer(inp: dict,
                                 transform=Optional[A.BasicTransform]
                                 ) -> Tuple[torch.Tensor, BoxList]:
    x, ann = inp
    x = np.array(x)
    if transform is not None:
        ann = transform(
            image=x, bboxes=ann['bboxes'], category_id=ann['category_id'])

    x = ann['image']
    b = torch.tensor(ann['bboxes'])
    c = ann['category_id']

    if len(b) == 0:
        y = BoxList(torch.empty((0, 4)), class_ids=torch.empty((0, )).long())
    else:
        boxes = torch.cat(
            [
                b[:, 1:2], b[:, 0:1], b[:, 1:2] + b[:, 3:4],
                b[:, 0:1] + b[:, 2:3]
            ],
            dim=1)
        class_ids = torch.tensor(c)
        y = BoxList(boxes, class_ids=class_ids)

    if np.issubdtype(x.dtype, np.unsignedinteger):
        max_val = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / max_val
    x = torch.from_numpy(x).permute(2, 0, 1).float()
    return x, y


def semantic_segmentation_transformer(inp: Tuple[Any, Any],
                                      transform=Optional[A.BasicTransform]
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    x, y = inp
    x, y = np.array(x), np.array(y)
    if transform is not None:
        out = transform(image=x, mask=y)
        x, y = out['image'], out['mask']
    return x, y


TF_TYPE_TO_TF_FUNC: Dict[TransformType, Callable] = {
    TransformType.noop: lambda x: x,
    TransformType.classification: classification_transformer,
    TransformType.regression: regression_transformer,
    TransformType.object_detection: object_detection_transformer,
    TransformType.semantic_segmentation: semantic_segmentation_transformer
}
