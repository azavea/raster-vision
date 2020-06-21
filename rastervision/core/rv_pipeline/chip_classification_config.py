from rastervision.pipeline.config import register_config, ConfigError
from rastervision.core.rv_pipeline import RVPipelineConfig
from rastervision.core.data.label_store import (
    ChipClassificationGeoJSONStoreConfig)
from rastervision.core.evaluation import ChipClassificationEvaluatorConfig


@register_config('chip_classification')
class ChipClassificationConfig(RVPipelineConfig):
    def build(self, tmp_dir):
        from rastervision.core.rv_pipeline.chip_classification import ChipClassification
        return ChipClassification(self, tmp_dir)

    def validate_config(self):
        if self.train_chip_sz != self.predict_chip_sz:
            raise ConfigError(
                'train_chip_sz must be equal to predict_chip_sz for chip '
                'classification.')

    def get_default_label_store(self, scene):
        return ChipClassificationGeoJSONStoreConfig()

    def get_default_evaluator(self):
        return ChipClassificationEvaluatorConfig()
