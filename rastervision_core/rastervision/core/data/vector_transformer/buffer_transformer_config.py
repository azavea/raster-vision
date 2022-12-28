from typing import TYPE_CHECKING, Dict, Optional

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.vector_transformer import (VectorTransformerConfig,
                                                       BufferTransformer)

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig


@register_config('buffer_transformer')
class BufferTransformerConfig(VectorTransformerConfig):
    """Configure a :class:`.BufferTransformer`.

    This is useful, for example, for buffering lines representing roads so that
    their width roughly matches the width of roads in the imagery.
    """
    geom_type: str = Field(
        ...,
        description='The geometry type to apply this transform to. '
        'E.g. "LineString", "Point", "Polygon".')
    class_bufs: Dict[int, Optional[float]] = Field(
        {},
        description='Mapping from class IDs to buffer amounts (in pixels). '
        'If a class ID is not found in the mapping, the value specified by '
        'the default_buf field will be used. If the buffer value for a class '
        'is None, then no buffering will be applied to the geoms of that '
        'class and the geom won\'t get converted to a Polygon. Not converting '
        'to Polygon is incompatible with the currently available '
        'LabelSources, but may be useful in the future.')
    default_buf: Optional[float] = Field(
        1,
        description='Default buffer to apply to classes not in class_bufs. '
        'If None, no buffering will be applied to the geoms of those classes.')

    def build(self, class_config: Optional['ClassConfig'] = None
              ) -> BufferTransformer:
        return BufferTransformer(
            self.geom_type,
            class_bufs=self.class_bufs,
            default_buf=self.default_buf)
