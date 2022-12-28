from rastervision.pipeline.config import Config, register_config


@register_config('label_store')
class LabelStoreConfig(Config):
    """Configure a :class:`.LabelStore`."""

    def build(self, class_config, crs_transformer, extent, tmp_dir):
        pass

    def update(self, pipeline=None, scene=None):
        pass
