from typing import TYPE_CHECKING
from pydantic.types import PositiveInt as PosInt

if TYPE_CHECKING:
    import numpy as np
    from rastervision.core.data.raster_transformer import RasterTransformer


def get_transformed_num_channels(
        raster_transformers: list['RasterTransformer'],
        in_channels: PosInt) -> PosInt:
    out_channels = in_channels
    for tf in raster_transformers:
        out_channels = tf.get_out_channels(out_channels)
    return out_channels


def get_transformed_dtype(raster_transformers: list['RasterTransformer'],
                          in_dtype: 'np.dtype') -> 'np.dtype':
    out_dtype = in_dtype
    for tf in raster_transformers:
        out_dtype = tf.get_out_dtype(out_dtype)
    return out_dtype
