from rastervision2.pipeline.config import Config, register_config


@register_config('label_store')
class LabelStoreConfig(Config):
    def build(self, class_config, crs_transformer):
        pass

    def update(self, pipeline=None, scene=None):
        pass
