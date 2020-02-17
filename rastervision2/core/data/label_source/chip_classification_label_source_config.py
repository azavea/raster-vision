from typing import Optional

from rastervision2.core.data.vector_source import (VectorSourceConfig)
from rastervision2.core.data.label_source import (
    LabelSourceConfig, ChipClassificationLabelSource)
from rastervision2.pipeline.config import (register_config, ConfigError)


@register_config('chip_classification_label_source')
class ChipClassificationLabelSourceConfig(LabelSourceConfig):
    vector_source: VectorSourceConfig
    ioa_thresh: Optional[float] = None
    use_intersection_over_cell: bool = False
    pick_min_class_id: bool = False
    background_class_id: Optional[int] = None
    infer_cells: bool = False
    cell_sz: Optional[int] = None

    def build(self, class_config, crs_transformer, extent=None):
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
