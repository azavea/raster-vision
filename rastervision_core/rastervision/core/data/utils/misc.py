from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union
import logging

import numpy as np
from PIL import ImageColor

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
    r = np.array(rgb_array[:, :, 0], dtype=np.uint32) * (1 << 16)
    g = np.array(rgb_array[:, :, 1], dtype=np.uint32) * (1 << 8)
    b = np.array(rgb_array[:, :, 2], dtype=np.uint32) * (1 << 0)
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


def match_extents(raster_source: 'RasterSource',
                  label_source: Union['LabelSource', 'LabelStore']) -> None:
    """Set ``label_souce`` extent equal to ``raster_source`` extent.

    Logs a warning if ``raster_source`` and ``label_source`` extents don't
    intersect when converted to map coordinates.

    Args:
        raster_source (RasterSource): Source of imagery for a scene.
        label_source (Union[LabelSource, LabelStore]): Source of labels for a
            scene. Can be a ``LabelStore``.
    """
    crs_tf_img = raster_source.crs_transformer
    crs_tf_label = label_source.crs_transformer
    extent_img_map = crs_tf_img.pixel_to_map(raster_source.extent)
    if label_source.extent is not None:
        extent_label_map = crs_tf_label.pixel_to_map(label_source.extent)
        if not extent_img_map.intersects(extent_label_map):
            rs_cls = type(raster_source).__name__
            ls_cls = type(label_source).__name__
            log.warning(f'{rs_cls} extent ({extent_img_map}) does '
                        f'not intersect with {ls_cls} extent '
                        f'({extent_label_map}).')
    # set LabelStore extent to RasterSource extent
    extent_label_pixel = crs_tf_label.map_to_pixel(extent_img_map)
    label_source.set_extent(extent_label_pixel)
