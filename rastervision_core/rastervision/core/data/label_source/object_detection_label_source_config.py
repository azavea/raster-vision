from typing import Optional

from rastervision.core.data.label_source import (LabelSourceConfig,
                                                 ObjectDetectionLabelSource)
from rastervision.core.data.vector_source import (VectorSourceConfig)
from rastervision.pipeline.config import (register_config, Field)


@register_config('object_detection_label_source')
class ObjectDetectionLabelSourceConfig(LabelSourceConfig):
    """Config for a read-only label source for object detection."""
    vector_source: VectorSourceConfig
    ioa_thresh: Optional[float] = Field(
        None,
        description='Minimum IOA of a polygon and cell for that polygon to be'
        'a candidate for setting the class_id.')
    clip: bool = Field(
        False,
        description='Clip bounding boxes to window limits when retrieving '
        'labels for a window.')

    def build(self, class_config, crs_transformer, extent, tmp_dir):
        vs = self.vector_source.build(class_config, crs_transformer)
        return ObjectDetectionLabelSource(
            vs, extent, ioa_thresh=self.ioa_thresh, clip=self.clip)
