from typing import TYPE_CHECKING

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.vector_transformer import (
    VectorTransformerConfig, ClassInferenceTransformer)

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig


@register_config('class_inference_transformer')
class ClassInferenceTransformerConfig(VectorTransformerConfig):
    """Configure a :class:`.ClassInferenceTransformer`."""

    default_class_id: int | None = Field(
        None,
        description='The default ``class_id`` to use if class cannot be '
        'inferred using other mechanisms. If a feature has an inferred '
        '``class_id`` of ``None``, then it will be deleted. '
        'Defaults to ``None``.')
    class_id_to_filter: dict[int, list] | None = Field(
        None,
        description='Map from ``class_id`` to JSON filter used to infer '
        'missing class IDs. Each key should be a class ID, and its value '
        'should be a boolean expression which is run against the property '
        'field for each feature. This allows matching different features to '
        'different class IDs based on its properties. The expression schema '
        'is that described by '
        'https://docs.mapbox.com/mapbox-gl-js/style-spec/other/#other-filter. '
        'Defaults to ``None``.')
    class_name_mapping: dict[str, str] | None = Field(
        None,
        description='``old_name --> new_name`` mapping for values in the '
        '``class_name`` or ``label`` property of the GeoJSON features. The '
        '``new_name`` must be a valid class name in the ``ClassConfig``. This '
        'can also be used to merge multiple classes into one e.g.: '
        '``dict(car="vehicle", truck="vehicle")``. Defaults to None.')

    def build(self, class_config: 'ClassConfig | None' = None
              ) -> ClassInferenceTransformer:
        return ClassInferenceTransformer(
            self.default_class_id,
            class_config=class_config,
            class_id_to_filter=self.class_id_to_filter,
            class_name_mapping=self.class_name_mapping)
