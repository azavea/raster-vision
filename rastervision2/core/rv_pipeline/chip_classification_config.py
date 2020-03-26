from rastervision2.pipeline.config import register_config
from rastervision2.core.rv_pipeline import RVPipelineConfig
from rastervision2.core.data.label_store import (
    ChipClassificationGeoJSONStoreConfig)
from rastervision2.core.evaluation import ChipClassificationEvaluatorConfig


@register_config('chip_classification')
class ChipClassificationConfig(RVPipelineConfig):
    def build(self, tmp_dir):
        from rastervision2.core.rv_pipeline.chip_classification import ChipClassification
        return ChipClassification(self, tmp_dir)

    def get_default_label_store(self, scene):
        return ChipClassificationGeoJSONStoreConfig()

    def get_default_evaluator(self):
        return ChipClassificationEvaluatorConfig()
