from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from rastervision.pytorch_learner.object_detection_utils import BoxList

if TYPE_CHECKING:
    from torch import Tensor
    from rastervision.core.box import Box


def chip_collate_fn_ss(batch: List[Tuple[Tuple[np.ndarray, np.ndarray], 'Box']]
                       ) -> Tuple[Tuple[np.ndarray, np.ndarray], List['Box']]:
    xs = np.stack([x for (x, _), _ in batch])
    ys = [None if np.isnan(y).all() else y for (_, y), _ in batch]
    ws = [w for (_, _), w in batch]
    return (xs, ys), ws


def chip_collate_fn_cc(batch: List[Tuple[Tuple[np.ndarray, np.ndarray], 'Box']]
                       ) -> Tuple[Tuple[np.ndarray, np.ndarray], List['Box']]:
    xs = np.stack([x for (x, _), _ in batch])
    ys = [None if np.isnan(y).all() else y for (_, y), _ in batch]
    ws = [w for (_, _), w in batch]
    return (xs, ys), ws


def chip_collate_fn_od(batch: List[Tuple[Tuple['Tensor', 'BoxList'], 'Box']]
                       ) -> Tuple[Tuple[np.ndarray, 'BoxList'], List['Box']]:
    xs = np.stack([x.numpy() for (x, _), _ in batch])
    # (..., c, h, w) --> (..., h, w, c)
    xs = xs.swapaxes(-3, -2).swapaxes(-2, -1)
    xs = (xs * 255).astype(np.uint8)
    ys = [y if isinstance(y, BoxList) else None for (_, y), _ in batch]
    ws = [w for (_, _), w in batch]
    return (xs, ys), ws
