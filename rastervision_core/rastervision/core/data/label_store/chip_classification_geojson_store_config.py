from typing import Optional
from os.path import join

from rastervision.core.data.label_store import (LabelStoreConfig,
                                                ChipClassificationGeoJSONStore)
from rastervision.pipeline.config import register_config, Field


@register_config('chip_classification_geojson_store')
class ChipClassificationGeoJSONStoreConfig(LabelStoreConfig):
    """Configure a :class:`.ChipClassificationGeoJSONStore`."""

    uri: Optional[str] = Field(
        None,
        description=
        ('URI of GeoJSON file with predictions. If None, and this Config is part of '
         'a SceneConfig inside an RVPipelineConfig, it will be auto-generated.'
         ))

    def build(self, class_config, crs_transformer, extent=None, tmp_dir=None):
        return ChipClassificationGeoJSONStore(self.uri, class_config,
                                              crs_transformer)

    def update(self, pipeline=None, scene=None):
        if self.uri is None and pipeline is not None and scene is not None:
            self.uri = join(pipeline.predict_uri, '{}.json'.format(scene.id))
