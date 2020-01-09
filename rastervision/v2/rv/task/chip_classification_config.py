from rastervision.v2.core.config import register_config
from rastervision.v2.rv.task import TaskConfig
from rastervision.v2.rv.data.label_store import (
    ChipClassificationGeoJSONStoreConfig)
from rastervision.v2.rv.evaluation import ChipClassificationEvaluatorConfig

@register_config('chip_classification')
class ChipClassificationConfig(TaskConfig):
    def build(self, tmp_dir):
        from rastervision.v2.rv.task.chip_classification import ChipClassification
        return ChipClassification(self, tmp_dir)

    def get_default_label_store(self, scene):
        return ChipClassificationGeoJSONStoreConfig()

    def get_default_evaluator(self):
        return ChipClassificationEvaluatorConfig()
