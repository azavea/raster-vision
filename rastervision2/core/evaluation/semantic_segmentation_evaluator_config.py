from typing import Optional
from os.path import join

from rastervision2.pipeline.config import register_config, Field
from rastervision2.core.evaluation.classification_evaluator_config import (
    ClassificationEvaluatorConfig)
from rastervision2.core.evaluation.semantic_segmentation_evaluator import (
    SemanticSegmentationEvaluator)


@register_config('semantic_segmentation_evaluator')
class SemanticSegmentationEvaluatorConfig(ClassificationEvaluatorConfig):
    vector_output_uri: Optional[str] = Field(
        None, description='URI of evaluation of vector output.')

    def build(self, class_config):
        return SemanticSegmentationEvaluator(class_config, self.output_uri,
                                             self.vector_output_uri)

    def update(self, pipeline=None):
        super().update(pipeline)

        if pipeline is not None and self.vector_output_uri is None:
            self.vector_output_uri = join(pipeline.eval_uri,
                                          'vector-eval.json')
