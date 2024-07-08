from rastervision.core.data.raster_source import (RasterSourceConfig,
                                                  RasterizedSourceConfig)
from rastervision.core.data.label_source import (
    LabelSourceConfig, SemanticSegmentationLabelSource)
from rastervision.pipeline.config import (register_config, Field)


def ss_label_source_config_upgrader(cfg_dict: dict,
                                    version: int) -> dict:  # pragma: no cover
    if version == 3:
        # removed in version 4
        cfg_dict.pop('rgb_class_config', None)
    return cfg_dict


@register_config(
    'semantic_segmentation_label_source',
    upgrader=ss_label_source_config_upgrader)
class SemanticSegmentationLabelSourceConfig(LabelSourceConfig):
    """Configure a :class:`.SemanticSegmentationLabelSource`."""

    raster_source: RasterSourceConfig | RasterizedSourceConfig = Field(
        ..., description='The labels in the form of rasters.')

    def build(self, class_config, crs_transformer, bbox=None,
              tmp_dir=None) -> SemanticSegmentationLabelSource:
        if isinstance(self.raster_source, RasterizedSourceConfig):
            rs = self.raster_source.build(class_config, crs_transformer, bbox)
        else:
            rs = self.raster_source.build(tmp_dir)
        return SemanticSegmentationLabelSource(rs, class_config)
