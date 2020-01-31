from rastervision.v2.core.config import Config

from rastervision.v2.core.config import register_config


@register_config('backend')
class BackendConfig(Config):
    def build(self, task, tmp_dir):
        raise NotImplementedError()

    def get_bundle_filenames(self):
        raise NotImplementedError()

    def update(self, task=None):
        pass
