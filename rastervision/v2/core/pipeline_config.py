from rastervision.v2.core.config import Config
from rastervision.v2.core.config import register_config
from rastervision.v2.core.pipeline import Pipeline


@register_config('pipeline')
class PipelineConfig(Config):
    root_uri: str
    rv_config: dict = None

    def build(self, tmp_dir):
        return Pipeline(self, tmp_dir)
