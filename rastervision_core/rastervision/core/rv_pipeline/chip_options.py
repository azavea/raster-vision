from typing import (TYPE_CHECKING, Any, Literal)
from enum import Enum

from pydantic import NonNegativeInt as NonNegInt, PositiveInt as PosInt
import numpy as np

from rastervision.core.rv_pipeline.utils import nodata_below_threshold
from rastervision.core.utils import Proportion
from rastervision.pipeline.config import (Config, ConfigError, Field,
                                          register_config, model_validator)

if TYPE_CHECKING:
    from typing import Self


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
    size: PosInt | tuple[PosInt, PosInt] = Field(
        ...,
        description='If method = sliding, this is the size of sliding window. '
        'If method = random, this is the size that all the windows are '
        'resized to before they are returned. If method = random and neither '
        'size_lims nor h_lims and w_lims have been specified, then size_lims '
        'is set to (size, size + 1).')
    stride: PosInt | tuple[PosInt, PosInt] | None = Field(
        None,
        description='Stride of sliding window. Only used if method = sliding.')
    padding: NonNegInt | tuple[NonNegInt, NonNegInt] | None = Field(
        None,
        description='How many pixels are windows allowed to overflow '
        'the edges of the raster source.')
    pad_direction: Literal['both', 'start', 'end'] = Field(
        'end',
        description='If "end", only pad ymax and xmax (bottom and right). '
        'If "start", only pad ymin and xmin (top and left). If "both", '
        'pad all sides. Has no effect if padding is zero. Defaults to "end".')
    size_lims: tuple[PosInt, PosInt] | None = Field(
        None,
        description='[min, max) interval from which window sizes will be '
        'uniformly randomly sampled. The upper limit is exclusive. To fix the '
        'size to a constant value, use size_lims = (sz, sz + 1). '
        'Only used if method = random. Specify either size_lims or '
        'h_lims and w_lims, but not both. If neither size_lims nor h_lims '
        'and w_lims have been specified, then this will be set to '
        '(size, size + 1).')
    h_lims: tuple[PosInt, PosInt] | None = Field(
        None,
        description='[min, max] interval from which window heights will be '
        'uniformly randomly sampled. Only used if method = random.')
    w_lims: tuple[PosInt, PosInt] | None = Field(
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

    @model_validator(mode='after')
    def validate_options(self) -> 'Self':
        method = self.method
        size = self.size
        if method == WindowSamplingMethod.sliding:
            if self.stride is None:
                self.stride = size
        elif method == WindowSamplingMethod.random:
            has_size_lims = self.size_lims is not None
            has_h_lims = self.h_lims is not None
            has_w_lims = self.w_lims is not None
            if not (has_size_lims or has_h_lims or has_w_lims):
                self.size_lims = (size, size + 1)
                has_size_lims = True
            if has_size_lims == (has_w_lims or has_h_lims):
                raise ConfigError('Specify either size_lims or h and w lims.')
            if has_h_lims != has_w_lims:
                raise ConfigError('h_lims and w_lims must both be specified')
        return self


@register_config('chip_options')
class ChipOptions(Config):
    """Configure the sampling and filtering of chips."""
    sampling: WindowSamplingConfig | dict[str, WindowSamplingConfig] = Field(
        ..., description='Window sampling config.')
    nodata_threshold: Proportion = Field(
        1.,
        description='Discard chips where the proportion of NODATA values is '
        'greater than or equal to this value. Might result in false positives '
        'if there are many legitimate black pixels in the chip. Use with '
        'caution. If 1.0, only chips that are fully NODATA will be discarded. '
        'Defaults to 1.0.')

    def get_chip_sz(self, scene_id: str | None = None) -> int:
        if isinstance(self.sampling, dict):
            if scene_id is None:
                raise KeyError(
                    'sampling is a dict[scene_id, WindowSamplingConfig], so '
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
