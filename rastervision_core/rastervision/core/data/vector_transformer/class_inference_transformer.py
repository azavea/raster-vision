from typing import TYPE_CHECKING, Dict, Optional
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
    """Infers missing class_ids from GeoJSON features.

        Rules:
            1) If class_id is in feature['properties'], use it.
            2) If class_config is set and class_name or label are in
                feature['properties'] and in class_config, use corresponding
                class_id.
            3) If class_id_to_filter is set and filter is true when applied to
                feature, use corresponding class_id.
            4) Otherwise, return the default_class_id
    """

    def __init__(self,
                 default_class_id: Optional[int],
                 class_config: Optional['ClassConfig'] = None,
                 class_id_to_filter: Optional[Dict[int, list]] = None):
        self.class_config = class_config
        self.class_id_to_filter = class_id_to_filter
        self.default_class_id = default_class_id

        if self.class_id_to_filter is not None:
            self.class_id_to_filter = {}
            for class_id, filter_exp in class_id_to_filter.items():
                self.class_id_to_filter[int(class_id)] = create_filter(
                    filter_exp)

    @staticmethod
    def infer_feature_class_id(
            feature: dict,
            default_class_id: Optional[int],
            class_config: Optional['ClassConfig'] = None,
            class_id_to_filter: Optional[Dict[int, list]] = None
    ) -> Optional[int]:
        """Infer the class_id for a GeoJSON feature.

        Rules:
            1) If class_id is in feature['properties'], use it.
            2) If class_config is set and class_name or label are in
                feature['properties'] and in class_config, use corresponding
                class_id.
            3) If class_id_to_filter is set and filter is true when applied to
                feature, use corresponding class_id.
            4) Otherwise, return the default_class_id.

        Args:
            feature (dict): GeoJSON feature.

        Returns:
            Optional[int]: Inferred class ID.
        """
        class_id = feature.get('properties', {}).get('class_id')
        if class_id is not None:
            return class_id

        if class_config is not None:
            class_name = feature.get('properties', {}).get('class_name')
            if class_name in class_config.names:
                return class_config.names.index(class_name)

            label = feature.get('properties', {}).get('label')
            if label in class_config.names:
                return class_config.names.index(label)

        if class_id_to_filter is not None:
            for class_id, filter_fn in class_id_to_filter.items():
                if filter_fn(feature):
                    return class_id

        return default_class_id

    def transform(self,
                  geojson: dict,
                  crs_transformer: Optional['CRSTransformer'] = None) -> dict:
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
                class_id_to_filter=self.class_id_to_filter)
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
