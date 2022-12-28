from typing import Union

from rastervision.core.data.raster_source import (RasterSourceConfig,
                                                  RasterizedSourceConfig)
from rastervision.core.data.label_source import (
    LabelSourceConfig, SemanticSegmentationLabelSource)
from rastervision.pipeline.config import (register_config, Field)


def ss_label_source_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version < 4:
        try:
            # removed in version 4
            del cfg_dict['rgb_class_config']
        except KeyError:
            pass
    return cfg_dict


@register_config(
    'semantic_segmentation_label_source',
    upgrader=ss_label_source_config_upgrader)
class SemanticSegmentationLabelSourceConfig(LabelSourceConfig):
    """Configure a :class:`.SemanticSegmentationLabelSource`."""

    raster_source: Union[RasterSourceConfig, RasterizedSourceConfig] = Field(
        ..., description='The labels in the form of rasters.')

    def build(self, class_config, crs_transformer, extent, tmp_dir):
        if isinstance(self.raster_source, RasterizedSourceConfig):
            rs = self.raster_source.build(class_config, crs_transformer,
                                          extent)
        else:
            rs = self.raster_source.build(tmp_dir)
        return SemanticSegmentationLabelSource(rs, class_config)
