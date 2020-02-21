from typing import Optional, Union

from rastervision2.core.data.raster_source import (RasterSourceConfig,
                                                   RasterizedSourceConfig)
from rastervision2.core.data.label_source import (
    LabelSourceConfig, SemanticSegmentationLabelSource)
from rastervision2.core.data.class_config import (ClassConfig)
from rastervision2.pipeline.config import (register_config)


@register_config('semantic_segmentation_label_source')
class SemanticSegmentationLabelSourceConfig(LabelSourceConfig):
    raster_source: Union[RasterSourceConfig, RasterizedSourceConfig]
    rgb_class_config: Optional[ClassConfig] = None

    def build(self, class_config, crs_transformer, extent, tmp_dir):
        if isinstance(self.raster_source, RasterizedSourceConfig):
            rs = self.raster_Source.build(class_config, crs_transformer,
                                          extent)
        else:
            rs = self.raster_source.build(tmp_dir)
        return SemanticSegmentationLabelSource(rs, self.rgb_class_config)
