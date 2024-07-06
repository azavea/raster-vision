from typing import TYPE_CHECKING, Iterable
from os.path import join

from rastervision.pipeline.config import register_config, Config, Field

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig
    from rastervision.core.evaluation import Evaluator
    from rastervision.core.rv_pipeline import RVPipelineConfig


@register_config('evaluator')
class EvaluatorConfig(Config):
    """Configure an :class:`.Evaluator`."""

    output_uri: str | None = Field(
        None,
        description='URI of directory where evaluator output will be saved. '
        'Evaluations for each scene-group will be save in a JSON file at '
        '<output_uri>/<scene-group-name>/eval.json. If None, and this Config '
        'is part of an RVPipeline, this field will be auto-generated.')

    def build(self,
              class_config: 'ClassConfig',
              scene_group: tuple[str, Iterable[str]] | None = None
              ) -> 'Evaluator':
        pass

    def get_output_uri(self, scene_group_name: str | None = None) -> str:
        if scene_group_name is None:
            return join(self.output_uri, 'eval.json')
        return join(self.output_uri, scene_group_name, 'eval.json')

    def update(self, pipeline: 'RVPipelineConfig | None' = None) -> None:
        if pipeline is not None and self.output_uri is None:
            self.output_uri = pipeline.eval_uri
