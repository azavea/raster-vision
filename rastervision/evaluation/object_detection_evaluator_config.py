import rastervision as rv
from rastervision.evaluation import ObjectDetectionEvaluator
from rastervision.evaluation \
    import (ClassificationEvaluatorConfig, ClassificationEvaluatorConfigBuilder)


class ObjectDetectionEvaluatorConfig(ClassificationEvaluatorConfig):
    def __init__(self, class_map, output_uri=None):
        super().__init__(rv.OBJECT_DETECTION_EVALUATOR, class_map, output_uri)

    def create_evaluator(self):
        return ObjectDetectionEvaluator(self.class_map, self.output_uri)


class ObjectDetectionEvaluatorConfigBuilder(
        ClassificationEvaluatorConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(ObjectDetectionEvaluatorConfig, prev)
