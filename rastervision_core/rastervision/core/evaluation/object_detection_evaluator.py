from rastervision.core.evaluation import (ClassificationEvaluator,
                                          ObjectDetectionEvaluation)


class ObjectDetectionEvaluator(ClassificationEvaluator):
    """Evaluates predictions for a set of scenes.
    """

    def __init__(self, class_config, output_uri):
        super().__init__(class_config, output_uri)

    def create_evaluation(self):
        return ObjectDetectionEvaluation(self.class_config)
