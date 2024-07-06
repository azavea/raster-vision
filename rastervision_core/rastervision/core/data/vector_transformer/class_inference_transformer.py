from typing import TYPE_CHECKING
from copy import deepcopy
import logging

from rastervision.core.data.vector_transformer import VectorTransformer
from rastervision.core.data.vector_transformer.label_maker.filter import (
    create_filter)
from rastervision.core.data.utils.geojson import features_to_geojson

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig, CRSTransformer

log = logging.getLogger(__name__)


class ClassInferenceTransformer(VectorTransformer):
    """Infers missing class IDs from GeoJSON features.

        Rules:
            1) If ``class_id`` is in ``feature['properties']``, use it.
            2) If ``class_config`` is set and ``"class_name"`` or ``"label"``
               are in ``feature['properties']`` and in ``class_config``, use
               corresponding ``class_id``.
            3) If ``class_id_to_filter`` is set and filter is true when applied
               to feature, use corresponding ``class_id``.
            4) Otherwise, return the ``default_class_id``.
    """

    def __init__(self,
                 default_class_id: int | None,
                 class_config: 'ClassConfig | None' = None,
                 class_id_to_filter: dict[int, list] | None = None,
                 class_name_mapping: dict[str, str] | None = None):
        """Constructor.

        Args:
            default_class_id: The default ``class_id`` to use if class cannot
                be inferred using other mechanisms. If a feature has an
                inferred ``class_id`` of None, then it will be deleted.
                Defaults to ``None``.
            class_config: ``ClassConfig`` to match the class names in the
                GeoJSON features to. Required if using ``class_name_mapping``.
                Defaults to None.
            class_id_to_filter: Map from ``class_id`` to JSON filter used to
                infer missing class IDs. Each key should be a class ID, and its
                value should be a boolean expression which is run against the
                property field for each feature. This allows matching different
                features to different class IDs based on its properties. The
                expression schema is that described by
                https://docs.mapbox.com/mapbox-gl-js/style-spec/other/#other-filter.
                Defaults to ``None``.
            class_name_mapping: ``old_name --> new_name`` mapping for values in
                the ``class_name`` or ``label`` property of the GeoJSON
                features. The ``new_name`` must be a valid class name in the
                ``ClassConfig``. This can also be used to merge multiple
                classes into one e.g.:
                ``dict(car="vehicle", truck="vehicle")``. Defaults to ``None``.
        """
        if class_name_mapping is not None and class_config is None:
            raise ValueError(
                'class_config must be specified if class_name_mapping is.')

        self.class_config = class_config
        self.class_id_to_filter = class_id_to_filter
        self.default_class_id = default_class_id
        self.class_name_mapping = class_name_mapping

        if self.class_id_to_filter is not None:
            self.class_id_to_filter = {}
            for class_id, filter_exp in class_id_to_filter.items():
                self.class_id_to_filter[int(class_id)] = create_filter(
                    filter_exp)

    @staticmethod
    def infer_feature_class_id(
            feature: dict,
            default_class_id: int | None,
            class_config: 'ClassConfig | None' = None,
            class_id_to_filter: dict[int, list] | None = None,
            class_name_mapping: dict[str, str] | None = None) -> int | None:
        """Infer the class ID for a GeoJSON feature.

        Rules:
            1) If ``class_id`` is in ``feature['properties']``, use it.
            2) If ``class_config`` is set and ``"class_name"`` or ``"label"``
               are in ``feature['properties']`` and in ``class_config``, use
               corresponding ``class_id``.
            3) If ``class_id_to_filter`` is set and filter is true when applied
               to feature, use corresponding ``class_id``.
            4) Otherwise, return the ``default_class_id``.

        Args:
            feature: GeoJSON feature.
            default_class_id: The default ``class_id`` to use if class cannot
                be inferred using other mechanisms. If a feature has an
                inferred ``class_id`` of None, then it will be deleted.
                Defaults to ``None``.
            class_config: ``ClassConfig`` to match the class names in the
                GeoJSON features to. Required if using ``class_name_mapping``.
                Defaults to None.
            class_id_to_filter: Map from ``class_id`` to JSON filter used to
                infer missing class IDs. Each key should be a class ID, and its
                value should be a boolean expression which is run against the
                property field for each feature. This allows matching different
                features to different class IDs based on its properties. The
                expression schema is that described by
                https://docs.mapbox.com/mapbox-gl-js/style-spec/other/#other-filter.
                Defaults to ``None``.
            class_name_mapping: ``old_name --> new_name`` mapping for values in
                the ``class_name`` or ``label`` property of the GeoJSON
                features. The ``new_name`` must be a valid class name in the
                ``ClassConfig``. This can also be used to merge multiple
                classes into one e.g.:
                ``dict(car="vehicle", truck="vehicle")``. Defaults to ``None``.

        Returns:
            int | None: Inferred class ID.
        """
        if class_name_mapping is not None and class_config is None:
            raise ValueError(
                'class_config must be specified if class_name_mapping is.')

        properties: dict = feature.get('properties', {})

        class_id = properties.get('class_id')
        if class_id is not None:
            return class_id

        if class_config is not None:
            if class_name_mapping is None:
                class_name_mapping = {}
            class_name = properties.get('class_name')
            if class_name is None:
                class_name = properties.get('label')
            class_name = class_name_mapping.get(class_name, class_name)
            if class_name in class_config.names:
                return class_config.names.index(class_name)

        if class_id_to_filter is not None:
            for class_id, filter_fn in class_id_to_filter.items():
                if filter_fn(feature):
                    return class_id

        return default_class_id

    def transform(self,
                  geojson: dict,
                  crs_transformer: 'CRSTransformer | None' = None) -> dict:
        """Add class_id to feature properties and drop features with no class.

        For each feature in geojson, the class_id is inferred and is set into
        feature['properties']. If the class_id is None (because none of the
        rules apply and the default_class_id is None), the feature is dropped.
        """
        new_features = []
        warned = False
        for feature in geojson['features']:
            class_id = self.infer_feature_class_id(
                feature,
                default_class_id=self.default_class_id,
                class_config=self.class_config,
                class_id_to_filter=self.class_id_to_filter,
                class_name_mapping=self.class_name_mapping)
            if class_id is not None:
                feature = deepcopy(feature)
                properties = feature.get('properties', {})
                properties['class_id'] = class_id
                feature['properties'] = properties
                new_features.append(feature)
            elif not warned:
                log.warning(
                    'ClassInferenceTransformer is dropping vector features because '
                    'class_id cannot be inferred. To avoid this behavior, '
                    'set default_class_id to a non-None value in '
                    'ClassInferenceTransformer.')
                warned = True

        new_geojson = features_to_geojson(new_features)
        return new_geojson
