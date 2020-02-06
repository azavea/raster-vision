from rastervision2.pipeline.config import Config

from rastervision2.pipeline.config import register_config


@register_config('backend')
class BackendConfig(Config):
    def build(self, pipeline, tmp_dir):
        raise NotImplementedError()

    def get_bundle_filenames(self):
        raise NotImplementedError()

    def update(self, pipeline=None):
        pass
