from rastervision.evaluation import (ClassificationEvaluator,
                                     ObjectDetectionEvaluation)


class ObjectDetectionEvaluator(ClassificationEvaluator):
    """Evaluates predictions for a set of scenes.
    """

    def __init__(self, class_map, output_uri):
        super().__init__(class_map, output_uri)

    def create_evaluation(self):
        return ObjectDetectionEvaluation(self.class_map)
