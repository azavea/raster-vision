from rastervision.v2.core.config import register_config
from rastervision.v2.rv.evaluation.evaluator_config import EvaluatorConfig

@register_config('classification_evaluator')
class ClassificationEvaluatorConfig(EvaluatorConfig):
    pass
