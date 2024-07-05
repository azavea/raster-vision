from typing import Literal, TypeVar
import math

from pydantic.types import NonNegativeInt as NonNegInt, PositiveInt as PosInt

T = TypeVar('T')


def ensure_tuple(x: T, n: int = 2) -> tuple[T, ...]:
    """Convert to n-tuple if not already an n-tuple."""
    if isinstance(x, tuple):
        if len(x) != n:
            raise ValueError()
        return x
    return tuple([x] * n)


def calculate_required_padding(
        extent_sz: PosInt | tuple[PosInt, PosInt],
        chip_sz: PosInt | tuple[PosInt, PosInt],
        stride: PosInt | tuple[PosInt, PosInt],
        pad_direction: Literal['start', 'end', 'both'],
        crop_sz: NonNegInt | None = None) -> tuple[NonNegInt, NonNegInt]:
    """Calculate min padding to ensure sliding windows cover all pixels.

    Args:
        extent_sz: Extent size as (h, w) tuple.
        chip_sz: Chip size as (h, w) tuple.
        stride: Stride size as (h_step, w_step) tuple.
        pad_direction: One of: 'start', 'end', 'both'.
        crop_sz: When cropping out window edges during semantic segmentation
            prediction, pixels at the edges of the scene can be left with no
            prediction if there is not enough padding. When ``crop_sz`` is
            specified, the calculated padding takes this into account. Has no
            effect if zero. Defaults to ``None``.

    Returns:
        Padding as (h_pad, w_pad) tuple.
    """
    extent_sz: tuple[PosInt, PosInt] = ensure_tuple(extent_sz)
    chip_sz: tuple[PosInt, PosInt] = ensure_tuple(chip_sz)
    stride: tuple[PosInt, PosInt] = ensure_tuple(stride)

    img_h, img_w = extent_sz
    chip_h, chip_w = chip_sz
    stride_h, stride_w = stride

    if chip_h < stride_h or chip_w < stride_w:
        raise ValueError(
            f'chip_sz ({chip_sz}) cannot be less than stride ({stride}).')

    if crop_sz is not None and crop_sz > 0:
        if pad_direction != 'both':
            raise ValueError(
                'crop_sz is only supported with pad_direction="both"')
        cropped_chip_h = chip_h - 2 * crop_sz
        cropped_chip_w = chip_w - 2 * crop_sz
        if cropped_chip_h < stride_h or cropped_chip_w < stride_w:
            raise ValueError(
                f'Cropped chip size ({(cropped_chip_h, cropped_chip_w)}) '
                f'cannot be less than stride ({stride}).')
        h_padding, w_padding = calculate_required_padding(
            extent_sz,
            (cropped_chip_h, cropped_chip_w),
            stride,
            pad_direction=pad_direction,
            crop_sz=None,
        )
        h_padding += 2 * crop_sz
        w_padding += 2 * crop_sz
    else:
        if img_h > chip_h:
            num_strides = math.ceil((img_h - chip_h) / stride_h)
            max_val = chip_h + num_strides * stride_h
            h_padding = max_val - img_h
        else:
            h_padding = chip_h - img_h
        if img_w > chip_w:
            num_strides = math.ceil((img_w - chip_w) / stride_w)
            max_val = chip_w + num_strides * stride_w
            w_padding = max_val - img_w
        else:
            w_padding = chip_w - img_w
        if pad_direction == 'both':
            h_padding = math.ceil(h_padding / 2)
            w_padding = math.ceil(w_padding / 2)

    padding = (h_padding, w_padding)
    return padding
