from rastervision.v2.core.config import Config, register_config

@register_config('label_source')
class LabelSourceConfig(Config):
    pass

    def build(self, class_config, crs_transformer):
        raise NotImplementedError()
