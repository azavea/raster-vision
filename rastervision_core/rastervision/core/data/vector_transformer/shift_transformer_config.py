from typing import TYPE_CHECKING, Optional

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.vector_transformer import (VectorTransformerConfig,
                                                       ShiftTransformer)

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig


@register_config('shift_transformer')
class ShiftTransformerConfig(VectorTransformerConfig):
    """Configure a :class:`.ShiftTransformer`."""

    x_shift: float = Field(
        0.0,
        descriptions='Distance in meters to shift along the x-axis. '
        'Postive values shift eastward.')
    y_shift: float = Field(
        0.0,
        descriptions='Distance in meters to shift along the y-axis. '
        'Postive values shift northward.')
    round_pixels: bool = Field(
        True,
        descriptions='Whether to round shifted pixel values to integers.')

    def build(self, class_config: Optional['ClassConfig'] = None
              ) -> ShiftTransformer:
        return ShiftTransformer(
            x_shift=self.x_shift,
            y_shift=self.y_shift,
            round_pixels=self.round_pixels)
