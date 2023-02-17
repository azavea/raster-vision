from rastervision.core.evaluation import (ClassificationEvaluator,
                                          ObjectDetectionEvaluation)


class ObjectDetectionEvaluator(ClassificationEvaluator):
    """Evaluates predictions for a set of scenes."""

    def create_evaluation(self):
        return ObjectDetectionEvaluation(self.class_config)
