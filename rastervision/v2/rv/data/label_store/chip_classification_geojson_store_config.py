from typing import Optional
from os.path import join

from rastervision.v2.rv.data.label_store import (
    LabelStoreConfig, ChipClassificationGeoJSONStore)
from rastervision.v2.core.config import register_config


@register_config('chip_classification_geojson_store')
class ChipClassificationGeoJSONStoreConfig(LabelStoreConfig):
    uri: Optional[str] = None

    def build(self, class_config, crs_transformer):
        return ChipClassificationGeoJSONStore(self.uri, class_config,
                                              crs_transformer)

    def update(self, task=None, scene=None):
        if self.uri is None and task is not None and scene is not None:
            self.uri = join(task.predict_uri, '{}.json'.format(scene.id))
