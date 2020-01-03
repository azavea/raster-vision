from typing import Optional

from rastervision.v2.rv.data.vector_source import (
    VectorSourceConfig)
from rastervision.v2.rv.data.label_source import LabelSourceConfig

class ChipClassificationLabelSourceConfig(LabelSourceConfig):
    vector_source: VectorSourceConfig
    ioa_thresh: Optional[float] = None
    use_intersection_over_cell: bool = False
    pick_min_class_id: bool = False
    background_class_id: Optional[int] = None
    infer_cells: bool = False
    cell_size: Optional[int] = None
