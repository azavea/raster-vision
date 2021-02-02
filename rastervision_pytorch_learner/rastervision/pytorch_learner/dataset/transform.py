from typing import Optional, Tuple, Any, Callable, Dict
from pydantic import PositiveInt as PosInt
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
    """Apply transform to image only."""
    x, y = inp
    if y is None:
        y = [-1]
    x, y = np.array(x), np.array(y)
    if transform is not None:
        out = transform(image=x)
        x = out['image']
    y = y.astype(np.long)
    return x, y


def regression_transformer(inp: Tuple[Any, Any],
                           transform=Optional[A.BasicTransform]
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Apply transform to image only."""
    x, y = inp
    if y is None:
        y = [np.nan]
    x, y = np.array(x), np.array(y, dtype=np.float32)
    if transform is not None:
        out = transform(image=x)
        x = out['image']
    return x, y


def yxyx_to_albu(yxyx: np.ndarray,
                 img_size: Tuple[PosInt, PosInt]) -> np.ndarray:
    """Unnormalized [ymin, xmin, ymax, xmax] to Albumentations format i.e.
    normalized [ymin, xmin, ymax, xmax].
    """
    h, w = img_size
    ymin, xmin, ymax, xmax = yxyx.T
    ymin, ymax = ymin / h, ymax / h
    xmin, xmax = xmin / w, xmax / w

    xmin = np.clip(xmin, 0., 1., out=xmin)
    xmax = np.clip(xmax, 0., 1., out=xmax)
    ymin = np.clip(ymin, 0., 1., out=ymin)
    ymax = np.clip(ymax, 0., 1., out=ymax)

    xyxy = np.stack([xmin, ymin, xmax, ymax], axis=1).reshape((-1, 4))
    return xyxy


def xywh_to_albu(xywh: np.ndarray,
                 img_size: Tuple[PosInt, PosInt]) -> np.ndarray:
    """Unnormalized [xmin, ymin, w, h] to Albumentations format i.e.
    normalized [ymin, xmin, ymax, xmax].
    """
    h, w = img_size
    xmin, ymin, box_w, box_h = xywh.T
    ymin, box_h = ymin / h, box_h / h
    xmin, box_w = xmin / w, box_w / w
    xmin, ymin, xmax, ymax = xmin, ymin, xmin + box_w, ymin + box_h

    xmin = np.clip(xmin, 0., 1., out=xmin)
    xmax = np.clip(xmax, 0., 1., out=xmax)
    ymin = np.clip(ymin, 0., 1., out=ymin)
    ymax = np.clip(ymax, 0., 1., out=ymax)

    xyxy = np.stack([xmin, ymin, xmax, ymax], axis=1).reshape((-1, 4))
    return xyxy


def albu_to_yxyx(xyxy: np.ndarray,
                 img_size: Tuple[PosInt, PosInt]) -> np.ndarray:
    """Albumentations format (i.e. normalized [ymin, xmin, ymax, xmax]) to
    unnormalized [ymin, xmin, ymax, xmax].
    """
    h, w = img_size
    xmin, ymin, xmax, ymax = xyxy.T
    xmin, ymin, xmax, ymax = xmin * w, ymin * h, xmax * w, ymax * h

    xmin = np.clip(xmin, 0., w, out=xmin)
    xmax = np.clip(xmax, 0., w, out=xmax)
    ymin = np.clip(ymin, 0., h, out=ymin)
    ymax = np.clip(ymax, 0., h, out=ymax)

    yxyx = np.stack([ymin, xmin, ymax, xmax], axis=1).reshape((-1, 4))
    return yxyx


def object_detection_transformer(
        inp: Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, str]],
        transform: Optional[A.BasicTransform] = None
) -> Tuple[torch.Tensor, BoxList]:
    """Apply transform to image, bounding boxes, and labels. Also perform
    normalization and conversion to pytorch tensors.

    The transform's BBoxParams are expected to have the format
    'albumentations' (i.e. normalized [ymin, xmin, ymax, xmax]).

    Args:
        inp (Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, str]]): Tuple of
            the form: (image, (boxes, class_ids, box_format)). box_format must
            be 'yxyx' or 'xywh'.
        transform (Optional[A.BasicTransform], optional): A transform.
            Defaults to None.

    Raises:
        NotImplementedError: If box_format is not 'yxyx' or 'xywh'.

    Returns:
        Tuple[torch.Tensor, BoxList]: Transformed image and boxes.
    """
    x, y = inp
    if y is not None:
        boxes, class_ids, box_format = y
    else:
        boxes, class_ids, box_format = np.empty((0, 4)), [], None

    img_size = x.shape[:2]
    if transform is not None:
        # The albumentations transform expects the bboxes to be in the
        # Albumentations format i.e. [ymin, xmin, ymax, xmax], so we convert to
        # that format before applying the transform.
        if box_format == 'yxyx':  # used by ObjectDetectionGeoDataset
            boxes = yxyx_to_albu(boxes, img_size)
        elif box_format == 'xywh':  # used by ObjectDetectionImageDataset
            boxes = xywh_to_albu(boxes, img_size)
        elif box_format is None:
            pass
        else:
            raise NotImplementedError(f'Unknown box_format: {box_format}.')

        out = transform(image=x, bboxes=boxes, category_id=class_ids)
        x = out['image']
        boxes = np.array(out['bboxes']).reshape((-1, 4))
        class_ids = np.array(out['category_id'])
        if len(boxes) > 0:
            boxes = albu_to_yxyx(boxes, img_size)

    # normalize x
    if np.issubdtype(x.dtype, np.unsignedinteger):
        max_val = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / max_val

    # convert to pytorch
    x = torch.from_numpy(x).permute(2, 0, 1).float()
    boxes = torch.from_numpy(boxes).float()
    class_ids = torch.from_numpy(class_ids).long()

    if len(boxes) == 0:
        boxes = torch.empty((0, 4)).float()

    y = BoxList(boxes, class_ids=class_ids)

    return x, y


def semantic_segmentation_transformer(inp: Tuple[Any, Any],
                                      transform=Optional[A.BasicTransform]
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Apply transform to image and mask."""
    x, y = inp
    x = np.array(x)

    if y is not None:
        y = np.array(y)
    else:
        y = np.full((1, 1), fill_value=-1, dtype=np.long)

    if transform is not None:
        if y is not None:
            out = transform(image=x, mask=y)
            x, y = out['image'], out['mask']
        else:
            out = transform(image=x)
            x = out['image']
    y = y.astype(np.long)
    return x, y


TF_TYPE_TO_TF_FUNC: Dict[TransformType, Callable] = {
    TransformType.noop: lambda x: x,
    TransformType.classification: classification_transformer,
    TransformType.regression: regression_transformer,
    TransformType.object_detection: object_detection_transformer,
    TransformType.semantic_segmentation: semantic_segmentation_transformer
}
