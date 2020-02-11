from os.path import join

from rastervision2.pipeline.config import Config
from rastervision2.pipeline.config import register_config
from rastervision2.pipeline.pipeline import Pipeline


@register_config('pipeline')
class PipelineConfig(Config):
    root_uri: str
    rv_config: dict = None

    def get_config_uri(self):
        return join(self.root_uri, 'pipeline.json')

    def build(self, tmp_dir):
        return Pipeline(self, tmp_dir)
