from rastervision.pipeline.config import register_config
from rastervision.core.evaluation.classification_evaluator_config import (
    ClassificationEvaluatorConfig)
from rastervision.core.evaluation.object_detection_evaluator import (
    ObjectDetectionEvaluator)


@register_config('object_detection_evaluator')
class ObjectDetectionEvaluatorConfig(ClassificationEvaluatorConfig):
    def build(self, class_config):
        return ObjectDetectionEvaluator(class_config, self.output_uri)
