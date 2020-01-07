from rastervision.v2.core.config import register_config
from rastervision.v2.rv.task import TaskConfig
from rastervision.v2.rv.data.label_store import (
    ChipClassificationGeoJSONStoreConfig)

@register_config('chip_classification')
class ChipClassificationConfig(TaskConfig):
    def get_pipeline(self):
        from rastervision.v2.rv.task.chip_classification import ChipClassification
        return ChipClassification

    def get_default_label_store(self, scene):
        return ChipClassificationGeoJSONStoreConfig()
