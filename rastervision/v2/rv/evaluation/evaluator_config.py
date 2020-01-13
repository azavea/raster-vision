from typing import Optional
from os.path import join

from rastervision.v2.core.config import register_config, Config


@register_config('evaluator')
class EvaluatorConfig(Config):
    output_uri: Optional[str] = None

    def update(self, task=None):
        if task is not None and self.output_uri is None:
            self.output_uri = join(task.eval_uri, 'eval.json')
