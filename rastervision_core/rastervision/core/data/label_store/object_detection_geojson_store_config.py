from typing import Optional
from os.path import join

from rastervision.core.data.label_store import (LabelStoreConfig,
                                                ObjectDetectionGeoJSONStore)
from rastervision.pipeline.config import register_config, Field


@register_config('object_detection_geojson_store')
class ObjectDetectionGeoJSONStoreConfig(LabelStoreConfig):
    """Configure an :class:`.ObjectDetectionGeoJSONStore`."""

    uri: Optional[str] = Field(
        None,
        description=
        ('URI of GeoJSON file with predictions. If None, and this Config is part of '
         'a SceneConfig inside an RVPipelineConfig, it will be auto-generated.'
         ))

    def build(self, class_config, crs_transformer, extent=None, tmp_dir=None):
        return ObjectDetectionGeoJSONStore(self.uri, class_config,
                                           crs_transformer)

    def update(self, pipeline=None, scene=None):
        if pipeline is not None and scene is not None:
            if self.uri is None:
                self.uri = join(pipeline.predict_uri,
                                '{}.json'.format(scene.id))
