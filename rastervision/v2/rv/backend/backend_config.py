from rastervision.v2.core.config import Config

from rastervision.v2.core.config import register_config

@register_config('backend')
class BackendConfig(Config):
    def update(self, parent=None):
        pass
    
    def build(self, tmp_dir):
        raise NotImplementedError()
