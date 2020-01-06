from rastervision.v2.rv.data.label_store import (
    LabelStoreConfig, ChipClassificationGeoJSONStore)
from rastervision.v2.core.config import register_config

@register_config('chip_classification_geojson_store')
class ChipClassificationGeoJSONStoreConfig(LabelStoreConfig):
    uri: str

    def build(self, class_config, crs_transformer):
        return ChipClassificationGeoJSONStore(self.uri, class_config, crs_transformer)