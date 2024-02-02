from typing import TYPE_CHECKING, Iterable, Optional, Tuple

from rastervision.pipeline.config import register_config
from rastervision.core.evaluation.classification_evaluator_config import (
    ClassificationEvaluatorConfig)
from rastervision.core.evaluation.semantic_segmentation_evaluator import (
    SemanticSegmentationEvaluator)

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig


def ss_evaluator_config_upgrader(cfg_dict: dict,
                                 version: int) -> dict:  # pragma: no cover
    if version == 2:
        # removed in version 3
        cfg_dict.pop('vector_output_uri', None)
    return cfg_dict


@register_config(
    'semantic_segmentation_evaluator', upgrader=ss_evaluator_config_upgrader)
class SemanticSegmentationEvaluatorConfig(ClassificationEvaluatorConfig):
    """Configure a :class:`.SemanticSegmentationEvaluator`."""

    def build(self,
              class_config: 'ClassConfig',
              scene_group: Optional[Tuple[str, Iterable[str]]] = None
              ) -> SemanticSegmentationEvaluator:
        if scene_group is None:
            output_uri = self.get_output_uri()
        else:
            group_name, _ = scene_group
            output_uri = self.get_output_uri(group_name)

        evaluator = SemanticSegmentationEvaluator(class_config, output_uri)
        return evaluator
