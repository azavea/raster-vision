import numpy as np

from rastervision.core.box import Box


def fill_overflow(bbox: Box,
                  window: Box,
                  chip: np.ndarray,
                  fill_value: int = 0) -> np.ndarray:
    """Where ``chip``'s ``window`` overflows bbox, fill with ``fill_value``.

    Args:
        bbox (Box): Bounding box.
        window (Box): Window corresponding to the ``chip``.
        chip (np.ndarray): (H, W, C) array.
        fill_value (int, optional): Value to set oveflowing pixels to.
            Defaults to 0.

    Returns:
        np.ndarray: Chip with overflowing regions filled with ``fill_value``.
    """
    top_overflow = max(0, bbox.ymin - window.ymin)
    bottom_overflow = max(0, window.ymax - bbox.ymax)
    left_overflow = max(0, bbox.xmin - window.xmin)
    right_overflow = max(0, window.xmax - bbox.xmax)

    *_, h, w, _ = chip.shape
    ymin, ymax = top_overflow, (h - bottom_overflow)
    xmin, xmax = left_overflow, (w - right_overflow)

    chip[..., :ymin, :, :] = fill_value
    chip[..., ymax:, :, :] = fill_value
    chip[..., :, :xmin, :] = fill_value
    chip[..., :, xmax:, :] = fill_value
    return chip


def pad_to_window_size(chip: np.ndarray, window: Box, bbox: Box,
                       fill_value: int) -> np.ndarray:
    """Where chip's window overflows bbox, pad chip with fill_value.

    Args:
        chip (np.ndarray): (H, W[, C]) array.
        bbox (Box): Bounding box.
        window (Box): Window corresponding to the ``chip``.
        fill_value (int, optional): Value to pad with. Defaults to 0.

    Returns:
        np.ndarray: Chip of size equal to window size with edges padded with
            fill_value where needed.
    """
    if window in bbox:
        return chip

    top_overflow = max(0, bbox.ymin - window.ymin)
    bottom_overflow = max(0, window.ymax - bbox.ymax)
    left_overflow = max(0, bbox.xmin - window.xmin)
    right_overflow = max(0, window.xmax - bbox.xmax)

    h, w = window.size
    ymin, ymax = top_overflow, (h - bottom_overflow)
    xmin, xmax = left_overflow, (w - right_overflow)

    if chip.ndim == 2:
        out_shape = (h, w)
        out = np.full(out_shape, fill_value)
        out[ymin:ymax, xmin:xmax] = chip
    else:
        *dims, _, _, c = chip.shape
        out_shape = (*dims, h, w, c)
        out = np.full(out_shape, fill_value)
        out[..., ymin:ymax, xmin:xmax, :] = chip
    return out
