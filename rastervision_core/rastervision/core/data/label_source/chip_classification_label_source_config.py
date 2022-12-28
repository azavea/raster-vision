from typing import Optional

from rastervision.core.data.vector_source import (VectorSourceConfig)
from rastervision.core.data.label_source import (LabelSourceConfig,
                                                 ChipClassificationLabelSource)
from rastervision.pipeline.config import (ConfigError, register_config, Field,
                                          validator, root_validator)
from rastervision.core.data.vector_transformer import (
    ClassInferenceTransformerConfig, BufferTransformerConfig)


def cc_label_source_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 4:
        # made non-optional in version 5
        cfg_dict['ioa_thresh'] = cfg_dict.get('ioa_thresh', 0.5)
    return cfg_dict


@register_config(
    'chip_classification_label_source',
    upgrader=cc_label_source_config_upgrader)
class ChipClassificationLabelSourceConfig(LabelSourceConfig):
    """Configure a :class:`.ChipClassificationLabelSource`.

    This can be provided explicitly as a grid of cells, or a grid of cells can
    be inferred from arbitrary polygons.
    """
    vector_source: Optional[VectorSourceConfig] = None
    ioa_thresh: float = Field(
        0.5,
        description=
        ('Minimum IOA of a polygon and cell for that polygon to be a candidate for '
         'setting the class_id.'))
    use_intersection_over_cell: bool = Field(
        False,
        description=
        ('If True, then use the area of the cell as the denominator in the IOA. '
         'Otherwise, use the area of the polygon.'))
    pick_min_class_id: bool = Field(
        False,
        description=
        ('If True, the class_id for a cell is the minimum class_id of the boxes in that '
         'cell. Otherwise, pick the class_id of the box covering the greatest area.'
         ))
    background_class_id: Optional[int] = Field(
        None,
        description=
        ('If not None, class_id to use as the background class; ie. the one that is used '
         'when a window contains no boxes. Cannot be None if infer_cells=True.'
         ))
    infer_cells: bool = Field(
        False,
        description='If True, infers a grid of cells based on the cell_sz.')
    cell_sz: Optional[int] = Field(
        None,
        description=
        ('Size of a cell to use in pixels. If None, and this Config is part '
         'of an RVPipeline, this field will be set from RVPipeline.train_chip_sz.'
         ))
    lazy: bool = Field(
        False,
        description='If True, labels will not be populated automatically '
        'during initialization of the label source.')

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

    @root_validator(skip_on_failure=True)
    def ensure_bg_class_id_if_inferring(cls, values: dict) -> dict:
        infer_cells = values.get('infer_cells')
        has_bg_class_id = values.get('background_class_id') is not None
        if infer_cells and not has_bg_class_id:
            raise ConfigError(
                'background_class_id is required if infer_cells=True.')
        return values

    def build(self, class_config, crs_transformer, extent=None, tmp_dir=None):
        if self.vector_source is None:
            raise ValueError('Cannot build with a None vector_source.')
        if self.infer_cells and self.cell_sz is None and not self.lazy:
            raise ValueError('Cannot build with infer_cells=True, '
                             'cell_sz=None and lazy=True.')
        vector_source = self.vector_source.build(class_config, crs_transformer)
        return ChipClassificationLabelSource(
            self, vector_source, extent=extent, lazy=self.lazy)

    def update(self, pipeline=None, scene=None):
        super().update(pipeline, scene)
        if self.cell_sz is None and pipeline is not None:
            self.cell_sz = pipeline.train_chip_sz
        if self.vector_source is not None:
            self.vector_source.update(pipeline, scene)
