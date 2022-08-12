from typing import TYPE_CHECKING, List, Optional

import numpy as np

from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.utils import (color_to_triple, color_to_integer,
                                          rgb_to_int_array)

if TYPE_CHECKING:
    from rastervision.core.data.class_config import ClassConfig


class RGBClassTransformer(RasterTransformer):
    """Maps RGB values to class IDs. Can also do the reverse."""

    def __init__(self, class_config: 'ClassConfig'):
        class_config.ensure_null_class()
        null_class_id = class_config.get_null_class_id()
        color_to_class = class_config.get_color_to_class_id()
        color_int_to_class = {
            color_to_integer(col): class_id
            for col, class_id in color_to_class.items()
        }
        class_to_color_triple = {
            class_id: color_to_triple(col)
            for col, class_id in color_to_class.items()
        }
        # i-th row of this array is the color-triple of the i-th class
        self.class_to_rgb_arr = np.array(
            [
                class_to_color_triple[c]
                for c in sorted(class_to_color_triple.keys())
            ],
            dtype=np.uint8)

        def color_int_to_class_fn(color: int) -> int:
            # Convert unspecified colors to null class
            return color_int_to_class.get(color, null_class_id)

        self.transform_color_int_to_class = np.vectorize(
            color_int_to_class_fn, otypes=[np.uint8])

    def transform(self,
                  chip: np.ndarray,
                  channel_order: Optional[List[int]] = None) -> np.ndarray:
        """Transform RGB array to array of class IDs or vice versa.

        Args:
            chip (np.ndarray): Numpy array of shape (H, W, 3).
            channel_order (Optional[List[int]], optional): List of indices of
                channels that were extracted from the raw imagery.
                Defaults to None.

        Returns:
            np.ndarray: An array of class IDs.
        """
        return self.rgb_to_class(chip)

    def rgb_to_class(self, rgb_labels: np.ndarray) -> np.ndarray:
        color_int_labels = rgb_to_int_array(rgb_labels)
        class_labels = self.transform_color_int_to_class(color_int_labels)
        return class_labels.astype(np.uint8)

    def class_to_rgb(self, class_labels: np.ndarray) -> np.ndarray:
        return self.class_to_rgb_arr[class_labels]
