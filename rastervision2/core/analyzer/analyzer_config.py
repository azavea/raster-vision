from rastervision2.pipeline.config import register_config, Config


@register_config('analyzer')
class AnalyzerConfig(Config):
    def build(self):
        pass

    def get_bundle_filenames(self):
        return []
