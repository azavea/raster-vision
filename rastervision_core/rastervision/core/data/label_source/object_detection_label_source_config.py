from rastervision.core.data.label_source import (LabelSourceConfig,
                                                 ObjectDetectionLabelSource)
from rastervision.core.data.vector_source import VectorSourceConfig
from rastervision.core.data.vector_transformer import (
    ClassInferenceTransformerConfig, BufferTransformerConfig)
from rastervision.pipeline.config import register_config, validator


@register_config('object_detection_label_source')
class ObjectDetectionLabelSourceConfig(LabelSourceConfig):
    """Configure an :class:`.ObjectDetectionLabelSource`."""

    vector_source: VectorSourceConfig

    @validator('vector_source')
    def ensure_required_transformers(
            cls, v: VectorSourceConfig) -> VectorSourceConfig:
        """Add class-inference and buffer transformers if absent."""
        tfs = v.transformers

        # add class inference transformer
        has_inf_tf = any(
            isinstance(tf, ClassInferenceTransformerConfig) for tf in tfs)
        if not has_inf_tf:
            tfs += [ClassInferenceTransformerConfig(default_class_id=None)]

        # add buffer transformers
        has_buf_tf = any(isinstance(tf, BufferTransformerConfig) for tf in tfs)
        if not has_buf_tf:
            tfs += [
                BufferTransformerConfig(geom_type='Point', default_buf=1),
                BufferTransformerConfig(geom_type='LineString', default_buf=1)
            ]

        return v

    def update(self, pipeline=None, scene=None):
        super().update(pipeline, scene)
        self.vector_source.update(pipeline, scene)

    def build(self, class_config, crs_transformer, extent,
              tmp_dir=None) -> ObjectDetectionLabelSource:
        vs = self.vector_source.build(class_config, crs_transformer)
        return ObjectDetectionLabelSource(vs, extent)
