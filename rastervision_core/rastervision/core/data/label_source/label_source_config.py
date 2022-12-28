from rastervision.pipeline.config import Config, register_config


@register_config('label_source')
class LabelSourceConfig(Config):
    """Configure a :class:`.LabelSource`."""

    def build(self, class_config, crs_transformer, extent, tmp_dir):
        raise NotImplementedError()

    def update(self, pipeline=None, scene=None):
        pass
