from rastervision.v2.rv.data.label_store import LabelStoreConfig
from rastervision.v2.core.config import register_config

@register_config('chip_classification_geojson_store')
class ChipClassificationGeoJSONStore(LabelStoreConfig):
    uri: str
