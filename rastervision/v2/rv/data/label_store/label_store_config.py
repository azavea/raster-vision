from rastervision.v2.core.config import Config, register_config

@register_config('label_store')
class LabelStoreConfig(Config):
    def build(self, class_config, crs_transformer):    
        pass

