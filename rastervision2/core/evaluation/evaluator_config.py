from typing import Optional
from os.path import join

from rastervision2.pipeline.config import register_config, Config


@register_config('evaluator')
class EvaluatorConfig(Config):
    output_uri: Optional[str] = None

    def update(self, pipeline=None):
        if pipeline is not None and self.output_uri is None:
            self.output_uri = join(pipeline.eval_uri, 'eval.json')
