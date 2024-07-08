from typing import TYPE_CHECKING, Iterable
from rastervision.pipeline.config import register_config
from rastervision.core.evaluation.classification_evaluator_config import (
    ClassificationEvaluatorConfig)
from rastervision.core.evaluation.chip_classification_evaluator import (
    ChipClassificationEvaluator)

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig


@register_config('chip_classification_evaluator')
class ChipClassificationEvaluatorConfig(ClassificationEvaluatorConfig):
    """Configure a :class:`.ChipClassificationEvaluator`."""

    def build(self,
              class_config: 'ClassConfig',
              scene_group: tuple[str, Iterable[str]] | None = None
              ) -> ChipClassificationEvaluator:
        if scene_group is None:
            output_uri = self.get_output_uri()
        else:
            group_name, _ = scene_group
            output_uri = self.get_output_uri(group_name)

        return ChipClassificationEvaluator(class_config, output_uri)
