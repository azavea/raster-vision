from rastervision2.pipeline.config import Config, register_config


@register_config('label_source')
class LabelSourceConfig(Config):
    def build(self, class_config, crs_transformer, extent):
        raise NotImplementedError()

    def update(self, task=None, scene=None):
        pass
