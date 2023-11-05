from rastervision.pipeline.config import register_config
from rastervision.core.evaluation.evaluator_config import EvaluatorConfig


@register_config('classification_evaluator')
class ClassificationEvaluatorConfig(EvaluatorConfig):
    """Configure a :class:`.ClassificationEvaluator`."""
