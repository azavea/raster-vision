from rastervision2.core.data.label_source import (LabelSourceConfig,
                                                  ObjectDetectionLabelSource)
from rastervision2.core.data.vector_source import (VectorSourceConfig)
from rastervision2.pipeline.config import (register_config)


@register_config('object_detection_label_source')
class ObjectDetectionLabelSourceConfig(LabelSourceConfig):
    """Config for a read-only label source for object detection."""
    vector_source: VectorSourceConfig

    def build(self, class_config, crs_transformer, extent, tmp_dir):
        vs = self.vector_source.build(class_config, crs_transformer)
        return ObjectDetectionLabelSource(vs, extent)
