from rastervision.evaluation import (ClassificationEvaluator,
                                     SemanticSegmentationEvaluation)


class SemanticSegmentationEvaluator(ClassificationEvaluator):
    """Evaluates predictions for a set of scenes.
    """

    def __init__(self, class_map, output_uri):
        super().__init__(class_map, output_uri)

    def create_evaluation(self):
        return SemanticSegmentationEvaluation(self.class_map)
