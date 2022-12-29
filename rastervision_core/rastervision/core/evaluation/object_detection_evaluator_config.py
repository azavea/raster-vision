from typing import TYPE_CHECKING, Iterable, Optional, Tuple

from rastervision.pipeline.config import register_config
from rastervision.core.evaluation.classification_evaluator_config import (
    ClassificationEvaluatorConfig)
from rastervision.core.evaluation.object_detection_evaluator import (
    ObjectDetectionEvaluator)

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig


@register_config('object_detection_evaluator')
class ObjectDetectionEvaluatorConfig(ClassificationEvaluatorConfig):
    """Configure an :class:`.ObjectDetectionEvaluator`."""

    def build(self,
              class_config: 'ClassConfig',
              scene_group: Optional[Tuple[str, Iterable[str]]] = None
              ) -> ObjectDetectionEvaluator:
        if scene_group is None:
            output_uri = self.get_output_uri()
        else:
            group_name, _ = scene_group
            output_uri = self.get_output_uri(group_name)

        return ObjectDetectionEvaluator(class_config, output_uri)
