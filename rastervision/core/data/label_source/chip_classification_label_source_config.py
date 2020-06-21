from typing import Optional

from rastervision.core.data.vector_source import (VectorSourceConfig)
from rastervision.core.data.label_source import (LabelSourceConfig,
                                                 ChipClassificationLabelSource)
from rastervision.pipeline.config import (register_config, ConfigError, Field)


@register_config('chip_classification_label_source')
class ChipClassificationLabelSourceConfig(LabelSourceConfig):
    """Config for a source of labels for chip classification.

    This can be provided explicitly as a grid of cells, or a grid of cells can be
    inferred from arbitrary polygons.
    """
    vector_source: VectorSourceConfig
    ioa_thresh: Optional[float] = Field(
        None,
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
         'when a window contains no boxes. If not set, empty windows have None set as '
         'their class_id which is considered a null value.'))
    infer_cells: bool = Field(
        False,
        description='If True, infers a grid of cells based on the cell_sz.')
    cell_sz: Optional[int] = Field(
        None,
        description=
        ('Size of a cell to use in pixels. If None, and this Config is part '
         'of an RVPipeline, this field will be set from RVPipeline.train_chip_sz.'
         ))

    def build(self, class_config, crs_transformer, extent=None, tmp_dir=None):
        vector_source = self.vector_source.build(class_config, crs_transformer)
        return ChipClassificationLabelSource(
            self, vector_source, class_config, crs_transformer, extent=extent)

    def update(self, pipeline=None, scene=None):
        super().update(pipeline, scene)
        if self.cell_sz is None and pipeline is not None:
            self.cell_sz = pipeline.train_chip_sz
        self.vector_source.update(pipeline, scene)

    def validate_config(self):
        if self.vector_source.has_null_class_bufs():
            raise ConfigError(
                'Setting buffer to None for a class in the vector_source is '
                'not allowed for ChipClassificationLabelSourceConfig.')
