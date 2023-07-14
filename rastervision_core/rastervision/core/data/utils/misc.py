from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import logging

import numpy as np
from PIL import ImageColor

from rastervision.core.box import Box

if TYPE_CHECKING:
    from rastervision.core.data import (RasterSource, LabelSource, LabelStore)

log = logging.getLogger(__name__)


def color_to_triple(
        color: Optional[Union[str, Sequence]] = None) -> Tuple[int, int, int]:
    """Given a PIL ImageColor string, return a triple of integers
    representing the red, green, and blue values.

    If color is None, return a random color.

    Args:
         color: A PIL ImageColor string

    Returns:
         An triple of integers

    """
    if color is None:
        r, g, b = np.random.randint(0, 256, size=3).tolist()
        return r, g, b
    elif isinstance(color, str):
        return ImageColor.getrgb(color)
    elif isinstance(color, (tuple, list)):
        return color
    else:
        raise TypeError(f'Unsupported type: {type(color)}')


def color_to_integer(color: str) -> int:
    """Given a PIL ImageColor string, return a packed integer.

    Args:
         color: A PIL ImageColor string

    Returns:
         An integer containing the packed RGB values.

    """
    triple = color_to_triple(color)
    r = triple[0] * (1 << 16)
    g = triple[1] * (1 << 8)
    b = triple[2] * (1 << 0)
    integer = r + g + b
    return integer


def normalize_color(
        color: Union[str, tuple, list]) -> Tuple[float, float, float]:
    """Convert color representation to a float 3-tuple with values in [0-1]."""
    if isinstance(color, str):
        color = color_to_triple(color)

    if isinstance(color, (tuple, list)):
        if all(isinstance(c, int) for c in color):
            return tuple(c / 255. for c in color)
        elif all(isinstance(c, float) for c in color):
            return tuple(color)
        else:
            raise ValueError('RGB values must be either all ints (0-255) '
                             'or all floats (0.0-1.0)')

    raise TypeError('Expected color to be a string or tuple or list, '
                    f'but found {type(color)}.')


def rgb_to_int_array(rgb_array: np.ndarray) -> np.ndarray:
    r = np.array(rgb_array[..., 0], dtype=np.uint32) * (1 << 16)
    g = np.array(rgb_array[..., 1], dtype=np.uint32) * (1 << 8)
    b = np.array(rgb_array[..., 2], dtype=np.uint32) * (1 << 0)
    return r + g + b


def all_equal(it: list):
    ''' Returns true if all elements are equal to each other '''
    return it.count(it[0]) == len(it)


def listify_uris(uris: Union[str, List[str]]) -> List[str]:
    """Convert to URI to list if needed."""
    if isinstance(uris, (list, tuple)):
        pass
    elif isinstance(uris, str):
        uris = [uris]
    else:
        raise TypeError(f'Expected str or List[str], but got {type(uris)}.')
    return uris


def match_bboxes(raster_source: 'RasterSource',
                 label_source: Union['LabelSource', 'LabelStore']) -> None:
    """Set ``label_souce`` bbox equal to ``raster_source`` bbox.

    Logs a warning if ``raster_source`` and ``label_source`` extents don't
    intersect when converted to map coordinates.

    Args:
        raster_source (RasterSource): Source of imagery for a scene.
        label_source (Union[LabelSource, LabelStore]): Source of labels for a
            scene. Can be a ``LabelStore``.
    """
    crs_tf_img = raster_source.crs_transformer
    crs_tf_label = label_source.crs_transformer
    bbox_img_map = crs_tf_img.pixel_to_map(raster_source.bbox)
    if label_source.bbox is not None:
        bbox_label_map = crs_tf_label.pixel_to_map(label_source.bbox)
        if not bbox_img_map.intersects(bbox_label_map):
            rs_cls = type(raster_source).__name__
            ls_cls = type(label_source).__name__
            log.warning(f'{rs_cls} bbox ({bbox_img_map}) does '
                        f'not intersect with {ls_cls} bbox '
                        f'({bbox_label_map}).')
    # set LabelStore bbox to RasterSource bbox
    bbox_label_pixel = crs_tf_label.map_to_pixel(bbox_img_map)
    label_source.set_bbox(bbox_label_pixel)


def parse_array_slices_2d(key: Union[tuple, slice],
                          extent: Box) -> Tuple[Box, List[Optional[Any]]]:
    """Parse 2D array-indexing inputs into a Box and slices."""
    return parse_array_slices_Nd(key, extent, dims=2, h_dim=0, w_dim=1)


def parse_array_slices_Nd(key: Union[tuple, slice],
                          extent: Box,
                          dims: int = 3,
                          h_dim: int = -3,
                          w_dim: int = -2) -> Tuple[Box, List[Optional[Any]]]:
    """Parse multi-dim array-indexing inputs into a Box and slices.

    Args:
        key (Union[tuple, slice]): Input to __getitem__.
        extent (Box): Extent of the raster/label source being indexed.
        dims (int): Total available indexable dims. Defaults to 3.
        h_dim (int): Index of height dim. Defaults to -3.
        w_dim (int): Index of width dim. Defaults to -2.

    Raises:
        NotImplementedError: If not (1 <= dims <= 3).
        TypeError: If key is not a slice or tuple.
        IndexError: if not (1 <= len(key) <= dims).
        TypeError: If the index for any of the dims is None.
        ValueError: If more than one Ellipsis ("...") in the input.
        ValueError: If h and w indices (first 2 dims) are not slices.
        NotImplementedError: If input contains negative values.

    Returns:
        Tuple[Box, list]: A Box representing the h and w slices and a list
            containing slices/index-values for all the dims.
    """
    if isinstance(key, slice):
        key = [key]
    elif isinstance(key, tuple):
        pass
    else:
        raise TypeError('Unsupported key type.')

    input_slices = list(key)

    if not (1 <= len(input_slices) <= dims):
        raise IndexError(f'Too many indices for {dims}-dimensional source.')
    if any(s is None for s in input_slices):
        raise TypeError('None is not a valid index.')

    if Ellipsis in input_slices:
        if input_slices.count(Ellipsis) > 1:
            raise ValueError('Only one ellipsis is allowed.')
        num_missing_dims = dims - (len(input_slices) - 1)
        filler_slices = [slice(None)] * num_missing_dims
        idx = input_slices.index(Ellipsis)
        # at the start
        if idx == 0:
            dim_slices = filler_slices + input_slices[(idx + 1):]
        # somewhere in the middle
        elif idx < (len(input_slices) - 1):
            dim_slices = (
                input_slices[:idx] + filler_slices + input_slices[(idx + 1):])
        # at the end
        else:
            dim_slices = input_slices[:idx] + filler_slices
    else:
        num_missing_dims = dims - len(input_slices)
        filler_slices = [slice(None)] * num_missing_dims
        dim_slices = input_slices + filler_slices

    if dim_slices[h_dim] is None:
        dim_slices[h_dim] = slice(None)
    if dim_slices[w_dim] is None:
        dim_slices[w_dim] = slice(None)
    h, w = dim_slices[h_dim], dim_slices[w_dim]
    if not (isinstance(h, slice) and isinstance(w, slice)):
        raise ValueError('h and w indices must be slices.')

    if any(x is not None and x < 0
           for x in [h.start, h.stop, h.step, w.start, w.stop, w.step]):
        raise NotImplementedError(
            'Negative indices are currently not supported.')

    # slices with missing endpoints get expanded to the extent limits
    H, W = extent.size
    _ymin = 0 if h.start is None else h.start
    _xmin = 0 if w.start is None else w.start
    _ymax = H if h.stop is None else h.stop
    _xmax = W if w.stop is None else w.stop
    window = Box(_ymin, _xmin, _ymax, _xmax)

    h_slice, w_slice = window.to_slices(h.step, w.step)
    dim_slices[h_dim] = h_slice
    dim_slices[w_dim] = w_slice

    return window, dim_slices


def ensure_json_serializable(obj: Any) -> dict:
    """Convert numpy types to JSON serializable equivalents."""
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(o) for o in obj]
    if isinstance(obj, np.ndarray):
        return ensure_json_serializable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, Box):
        return obj.to_dict()
    return obj
