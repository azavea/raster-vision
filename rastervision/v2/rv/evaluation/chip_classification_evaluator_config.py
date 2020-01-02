import rastervision as rv
from rastervision.v2.evaluation import ChipClassificationEvaluator
from rastervision.v2.evaluation \
    import (ClassificationEvaluatorConfig, ClassificationEvaluatorConfigBuilder)


class ChipClassificationEvaluatorConfig(ClassificationEvaluatorConfig):
    def __init__(self, class_map, output_uri=None):
        super().__init__(rv.CHIP_CLASSIFICATION_EVALUATOR, class_map,
                         output_uri)

    def create_evaluator(self):
        return ChipClassificationEvaluator(self.class_map, self.output_uri)


class ChipClassificationEvaluatorConfigBuilder(
        ClassificationEvaluatorConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(ChipClassificationEvaluatorConfig, prev)
