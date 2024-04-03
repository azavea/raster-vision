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


def apply_transform(transform: A.BasicTransform, **kwargs) -> dict:
    """Apply Albumentations transform to possibly batched images.

    In case of batched images, the same transform is applied to all of them.
    This is useful for when the images represent a time-series.

    Args:
        transform (A.BasicTransform): An albumentations transform.
        **kwargs: Extra args for ``transform``.

    Returns:
        dict: Output of ``transform``. If ndim == 4, the transformed image in
        the dict is also 4-dimensional.
    """
    img = kwargs['image']
    if img.ndim == 3:
        return transform(**kwargs)

    if img.ndim != 4:
        raise NotImplementedError(
            f'Image should have 3 or 4 dims. Found {img.ndim}.')

    batch_size = len(img)

    if len(transform._additional_targets) != (batch_size - 1):
        additional_targets = {f'img{i}': 'image' for i in range(1, batch_size)}
        transform.add_targets(additional_targets)

    img = kwargs.pop('image')
    img_keys = transform._additional_targets.keys()
    img_args = dict(zip(img_keys, img[1:]))
    out = transform(image=img[0], **kwargs, **img_args)
    out['image'] = np.stack([out.pop('image')] + [out[k] for k in img_keys])

    return out


def classification_transformer(inp: Tuple[np.ndarray, Optional[int]],
                               transform=Optional[A.BasicTransform]
                               ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply transform to image only."""
    x, y = inp
    x = np.array(x)
    if transform is not None:
        out = apply_transform(transform, image=x)
        x = out['image']
    if y is not None:
        y = np.array(y, dtype=int)
    return x, y


def regression_transformer(inp: Tuple[np.ndarray, Optional[Any]],
                           transform=Optional[A.BasicTransform]
                           ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply transform to image only."""
    x, y = inp
    x = np.array(x)
    if transform is not None:
        out = apply_transform(transform, image=x)
        x = out['image']
    if y is not None:
        y = np.array(y, dtype=float)
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
        inp: Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray, str]]],
        transform: Optional[A.BasicTransform] = None
) -> Tuple[torch.Tensor, Optional[BoxList]]:
    """Apply transform to image, bounding boxes, and labels. Also perform
    normalization and conversion to pytorch tensors.

    The transform's BBoxParams are expected to have the format
    'albumentations' (i.e. normalized [ymin, xmin, ymax, xmax]).

    Args:
        inp (Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray, str]]]):
            Tuple of the form: (image, (boxes, class_ids, box_format)).
            box_format must be 'yxyx' or 'xywh'.
        transform (Optional[A.BasicTransform], optional): A transform.
            Defaults to None.

    Raises:
        NotImplementedError: If box_format is not 'yxyx' or 'xywh'.

    Returns:
        Tuple[torch.Tensor, BoxList]: Transformed image and boxes.
    """
    x, y = inp
    img_size = x.shape[:2]

    if y is not None:
        boxes, class_ids, box_format = y

    if transform is not None:
        if y is None:
            x = apply_transform(
                transform, image=x, bboxes=[], category_id=[])['image']
        else:
            # The albumentations transform expects the bboxes to be in the
            # Albumentations format i.e. [ymin, xmin, ymax, xmax], so we convert to
            # that format before applying the transform.
            if box_format == 'yxyx':  # used by ObjectDetectionGeoDataset
                boxes = yxyx_to_albu(boxes, img_size)
            elif box_format == 'xywh':  # used by ObjectDetectionImageDataset
                boxes = xywh_to_albu(boxes, img_size)
            else:
                raise NotImplementedError(f'Unknown box_format: {box_format}.')

            out = apply_transform(
                transform, image=x, bboxes=boxes, category_id=class_ids)
            x = out['image']
            boxes = np.array(out['bboxes']).reshape((-1, 4))
            class_ids = np.array(out['category_id'])
            if len(boxes) > 0:
                boxes = albu_to_yxyx(boxes, x.shape[:2])
            new_box_format = 'yxyx'
    elif y is not None:
        new_box_format = box_format

    if y is not None:
        boxes = torch.from_numpy(boxes).float()
        class_ids = torch.from_numpy(class_ids).long()
        if len(boxes) == 0:
            boxes = torch.empty((0, 4)).float()
        y = BoxList(boxes, format=new_box_format, class_ids=class_ids)

    # normalize x
    if np.issubdtype(x.dtype, np.unsignedinteger):
        max_val = np.iinfo(x.dtype).max
        x = x.astype(float) / max_val

    # convert to pytorch
    x = torch.from_numpy(x).permute(2, 0, 1).float()

    return x, y


def semantic_segmentation_transformer(
        inp: Tuple[np.ndarray, Optional[np.ndarray]],
        transform=Optional[A.BasicTransform]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply transform to image and mask."""
    x, y = inp
    x = np.array(x)
    if transform is not None:
        if y is None:
            x = apply_transform(transform, image=x)['image']
        else:
            y = np.array(y)
            out = apply_transform(transform, image=x, mask=y)
            x, y = out['image'], out['mask']
    if y is not None:
        y = y.astype(int)
    return x, y


TF_TYPE_TO_TF_FUNC: Dict[TransformType, Callable] = {
    TransformType.noop: lambda x, tf: x,
    TransformType.classification: classification_transformer,
    TransformType.regression: regression_transformer,
    TransformType.object_detection: object_detection_transformer,
    TransformType.semantic_segmentation: semantic_segmentation_transformer
}
