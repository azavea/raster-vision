from typing import (Any, Dict, Literal, Optional, Tuple, Union)
from enum import Enum

from pydantic import (PositiveInt as PosInt, conint)
import numpy as np

from rastervision.core.utils.misc import Proportion
from rastervision.core.rv_pipeline.utils import nodata_below_threshold
from rastervision.pipeline.config import (Config, ConfigError, Field,
                                          register_config, root_validator)

NonNegInt = conint(ge=0)


class WindowSamplingMethod(Enum):
    """Enum for window sampling methods.

    Attributes:
        sliding: Sliding windows.
        random: Randomly sampled windows.
    """
    sliding = 'sliding'
    random = 'random'


@register_config('window_sampling')
class WindowSamplingConfig(Config):
    """Configure the sampling of chip windows."""

    method: WindowSamplingMethod = Field(
        WindowSamplingMethod.sliding, description='')
    size: Union[PosInt, Tuple[PosInt, PosInt]] = Field(
        ...,
        description='If method = sliding, this is the size of sliding window. '
        'If method = random, this is the size that all the windows are '
        'resized to before they are returned. If method = random and neither '
        'size_lims nor h_lims and w_lims have been specified, then size_lims '
        'is set to (size, size + 1).')
    stride: Optional[Union[PosInt, Tuple[PosInt, PosInt]]] = Field(
        None,
        description='Stride of sliding window. Only used if method = sliding.')
    padding: Optional[Union[NonNegInt, Tuple[NonNegInt, NonNegInt]]] = Field(
        None,
        description='How many pixels are windows allowed to overflow '
        'the edges of the raster source.')
    pad_direction: Literal['both', 'start', 'end'] = Field(
        'end',
        description='If "end", only pad ymax and xmax (bottom and right). '
        'If "start", only pad ymin and xmin (top and left). If "both", '
        'pad all sides. Has no effect if paddiong is zero. Defaults to "end".')
    size_lims: Optional[Tuple[PosInt, PosInt]] = Field(
        None,
        description='[min, max) interval from which window sizes will be '
        'uniformly randomly sampled. The upper limit is exclusive. To fix the '
        'size to a constant value, use size_lims = (sz, sz + 1). '
        'Only used if method = random. Specify either size_lims or '
        'h_lims and w_lims, but not both. If neither size_lims nor h_lims '
        'and w_lims have been specified, then this will be set to '
        '(size, size + 1).')
    h_lims: Optional[Tuple[PosInt, PosInt]] = Field(
        None,
        description='[min, max] interval from which window heights will be '
        'uniformly randomly sampled. Only used if method = random.')
    w_lims: Optional[Tuple[PosInt, PosInt]] = Field(
        None,
        description='[min, max] interval from which window widths will be '
        'uniformly randomly sampled. Only used if method = random.')
    max_windows: NonNegInt = Field(
        10_000,
        description='Max number of windows to sample. Only used if '
        'method = random.')
    max_sample_attempts: PosInt = Field(
        100,
        description='Max attempts when trying to find a window within the AOI '
        'of a scene. Only used if method = random and the scene has '
        'aoi_polygons specified.')
    efficient_aoi_sampling: bool = Field(
        True,
        description='If the scene has AOIs, sampling windows at random '
        'anywhere in the extent and then checking if they fall within any of '
        'the AOIs can be very inefficient. This flag enables the use of an '
        'alternate algorithm that only samples window locations inside the '
        'AOIs. Only used if method = random and the scene has aoi_polygons '
        'specified. Defaults to True')
    within_aoi: bool = Field(
        True,
        description='If True and if the scene has an AOI, only sample windows '
        'that lie fully within the AOI. If False, windows only partially '
        'intersecting the AOI will also be allowed.')

    @root_validator(skip_on_failure=True)
    def validate_options(cls, values: dict) -> dict:
        method = values.get('method')
        size = values.get('size')
        if method == WindowSamplingMethod.sliding:
            has_stride = values.get('stride') is not None

            if not has_stride:
                values['stride'] = size
        elif method == WindowSamplingMethod.random:
            size_lims = values.get('size_lims')
            h_lims = values.get('h_lims')
            w_lims = values.get('w_lims')

            has_size_lims = size_lims is not None
            has_h_lims = h_lims is not None
            has_w_lims = w_lims is not None

            if not (has_size_lims or has_h_lims or has_w_lims):
                size_lims = (size, size + 1)
                has_size_lims = True
                values['size_lims'] = size_lims
            if has_size_lims == (has_w_lims or has_h_lims):
                raise ConfigError('Specify either size_lims or h and w lims.')
            if has_h_lims != has_w_lims:
                raise ConfigError('h_lims and w_lims must both be specified')
        return values


@register_config('chip_options')
class ChipOptions(Config):
    """Configure the sampling and filtering of chips."""
    sampling: Union[WindowSamplingConfig, Dict[
        str, WindowSamplingConfig]] = Field(
            ..., description='Window sampling config.')
    nodata_threshold: Proportion = Field(
        1.,
        description='Discard chips where the proportion of NODATA values is '
        'greater than or equal to this value. Might result in false positives '
        'if there are many legitimate black pixels in the chip. Use with '
        'caution. If 1.0, only chips that are fully NODATA will be discarded. '
        'Defaults to 1.0.')

    def get_chip_sz(self, scene_id: Optional[str] = None) -> int:
        if isinstance(self.sampling, dict):
            if scene_id is None:
                raise KeyError(
                    'sampling is a Dict[scene_id, WindowSamplingConfig], so '
                    'there is no single chip size. Specify a valid scene_id '
                    'to get the chip size for a particular scene.')
            return self.sampling[scene_id].size
        return self.sampling.size

    def keep_chip(self, chip: np.ndarray, label: Any) -> bool:
        """Decide whether to keep or discard chip.

        Args:
            chip: Chip raster.
            label: Associated label.
        """
        out = nodata_below_threshold(chip, self.nodata_threshold)
        return out
