from rastervision.v2.core.config import Config
from rastervision.v2.core.config import register_config
from rastervision.v2.core.pipeline import Pipeline

@register_config('pipeline')
class PipelineConfig(Config):
    root_uri: str

    def get_pipeline(self):
        return Pipeline


def get_config(runner):
    return PipelineConfig(root_uri='/opt/data/')


def register_plugin(registry):
    pass
