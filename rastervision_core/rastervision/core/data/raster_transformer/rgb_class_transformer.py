from typing import TYPE_CHECKING, List, Optional

import numpy as np

from rastervision.core.data.raster_transformer import (RasterTransformer,
                                                       ReclassTransformer)
from rastervision.core.data.utils import (color_to_triple, color_to_integer,
                                          rgb_to_int_array)

if TYPE_CHECKING:
    from rastervision.core.data.class_config import ClassConfig


class RGBClassTransformer(RasterTransformer):
    """Maps RGB values to class IDs. Can also do the reverse."""

    def __init__(self, class_config: 'ClassConfig'):
        class_config.ensure_null_class()
        self.null_class_id = class_config.null_class_id
        color_to_class = class_config.get_color_to_class_id()

        self.rgb_int_to_class = {
            color_to_integer(col): class_id
            for col, class_id in color_to_class.items()
        }
        self.rgb_int_to_class_tf = ReclassTransformer(self.rgb_int_to_class)

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

    def rgb_to_class(self, array_rgb: np.ndarray) -> np.ndarray:
        array_int = rgb_to_int_array(array_rgb)
        array_class_id = self.rgb_int_to_class_tf.transform(array_int)
        return array_class_id.astype(np.uint8)

    def class_to_rgb(self, class_labels: np.ndarray) -> np.ndarray:
        return self.class_to_rgb_arr[class_labels]
