from rastervision.v2.core.config import register_config
from rastervision.v2.rv.evaluation.classification_evaluator_config import (
    ClassificationEvaluatorConfig)
from rastervision.v2.rv.evaluation.chip_classification_evaluator import (
    ChipClassificationEvaluator)


@register_config('chip_classification_evaluator')
class ChipClassificationEvaluatorConfig(ClassificationEvaluatorConfig):
    def build(self, class_config):
        return ChipClassificationEvaluator(class_config, self.output_uri)
