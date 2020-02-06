from typing import Optional
from os.path import join

from rastervision2.core.data.label_store import (
    LabelStoreConfig, ChipClassificationGeoJSONStore)
from rastervision2.pipeline.config import register_config


@register_config('chip_classification_geojson_store')
class ChipClassificationGeoJSONStoreConfig(LabelStoreConfig):
    uri: Optional[str] = None

    def build(self, class_config, crs_transformer):
        return ChipClassificationGeoJSONStore(self.uri, class_config,
                                              crs_transformer)

    def update(self, pipeline=None, scene=None):
        if self.uri is None and pipeline is not None and scene is not None:
            self.uri = join(pipeline.predict_uri, '{}.json'.format(scene.id))
