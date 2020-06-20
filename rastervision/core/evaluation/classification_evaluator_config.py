from rastervision2.pipeline.config import register_config
from rastervision2.core.evaluation.evaluator_config import EvaluatorConfig


@register_config('classification_evaluator')
class ClassificationEvaluatorConfig(EvaluatorConfig):
    pass
