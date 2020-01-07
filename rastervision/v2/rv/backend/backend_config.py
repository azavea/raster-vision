from rastervision.v2.core.config import Config

from rastervision.v2.core.config import register_config

@register_config('backend')
class BackendConfig(Config):
    def build(self, tmp_dir):
        raise NotImplementedError()

    def update(self, task=None):
        pass
