import rastervision as rv
from rastervision.evaluation import SemanticSegmentationEvaluator
from rastervision.evaluation \
    import (ClassificationEvaluatorConfig, ClassificationEvaluatorConfigBuilder)


class SemanticSegmentationEvaluatorConfig(ClassificationEvaluatorConfig):
    def __init__(self, class_map, output_uri=None):
        super().__init__(rv.SEMANTIC_SEGMENTATION_EVALUATOR, class_map,
                         output_uri)

    def create_evaluator(self):
        return SemanticSegmentationEvaluator(self.class_map, self.output_uri)


class SemanticSegmentationEvaluatorConfigBuilder(
        ClassificationEvaluatorConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(SemanticSegmentationEvaluatorConfig, prev)
