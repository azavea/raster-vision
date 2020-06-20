from rastervision2.pipeline.config import register_config
from rastervision2.core.evaluation.classification_evaluator_config import (
    ClassificationEvaluatorConfig)
from rastervision2.core.evaluation.chip_classification_evaluator import (
    ChipClassificationEvaluator)


@register_config('chip_classification_evaluator')
class ChipClassificationEvaluatorConfig(ClassificationEvaluatorConfig):
    def build(self, class_config):
        return ChipClassificationEvaluator(class_config, self.output_uri)
