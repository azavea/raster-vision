from typing import TYPE_CHECKING, Dict, Optional

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.vector_transformer import (
    VectorTransformerConfig, ClassInferenceTransformer)

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig


@register_config('class_inference_transformer')
class ClassInferenceTransformerConfig(VectorTransformerConfig):
    """Configure a :class:`.ClassInferenceTransformer`."""

    default_class_id: Optional[int] = Field(
        None,
        description='The default class_id to use if class cannot be inferred '
        'using other mechanisms. If a feature has an inferred class_id of '
        'None, then it will be deleted.')
    class_id_to_filter: Optional[Dict[int, list]] = Field(
        None,
        description='Map from class_id to JSON filter used to infer missing '
        'class_ids. Each key should be a class id, and its value should be a '
        'boolean expression which is run against the property field for each '
        'feature. This allows matching different features to different '
        'class IDs based on its properties. The expression schema is that '
        'described by '
        'https://docs.mapbox.com/mapbox-gl-js/style-spec/other/#other-filter')

    def build(self, class_config: Optional['ClassConfig'] = None
              ) -> ClassInferenceTransformer:
        return ClassInferenceTransformer(
            self.default_class_id,
            class_config=class_config,
            class_id_to_filter=self.class_id_to_filter)
