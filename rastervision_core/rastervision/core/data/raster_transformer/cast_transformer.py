from rastervision.core.data.raster_transformer.raster_transformer \
    import RasterTransformer
from rastervision.pipeline.utils import repr_with_args

import numpy as np


class CastTransformer(RasterTransformer):
    """Casts chips to the specified dtype."""

    def __init__(self, to_dtype: str):
        """Constructor.

        Args:
            to_dtype: (str) dtype to cast the chips to.
        """
        self.to_dtype = np.dtype(to_dtype)

    def __repr__(self):
        return repr_with_args(self, to_dtype=str(self.to_dtype))

    def transform(self, chip: np.ndarray) -> np.ndarray:
        """Cast chip to dtype ``self.to_dtype``.

        Args:
            chip: Array of shape (..., H, W, C).

        Returns:
            Array of shape (..., H, W, C)
        """
        return chip.astype(self.to_dtype)

    def get_out_dtype(self, in_dtype: 'np.dtype') -> 'np.dtype':
        return self.to_dtype
