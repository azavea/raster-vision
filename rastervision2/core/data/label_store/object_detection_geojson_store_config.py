from typing import Optional
from os.path import join

from rastervision2.core.data.label_store import (LabelStoreConfig,
                                                 ObjectDetectionGeoJSONStore)
from rastervision2.pipeline.config import register_config


@register_config('object_detection_geojson_store')
class ObjectDetectionGeoJSONStoreConfig(LabelStoreConfig):
    uri: Optional[str] = None

    def build(self, class_config, crs_transformer, extent=None, tmp_dir=None):
        return ObjectDetectionGeoJSONStore(self.uri, class_config,
                                           crs_transformer)

    def update(self, pipeline=None, scene=None):
        if pipeline is not None and scene is not None:
            if self.uri is None:
                self.uri = join(pipeline.predict_uri,
                                '{}.json'.format(scene.id))
