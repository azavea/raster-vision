from rastervision.new_version.pipeline.config import Config


class PipelineConfig(Config):
    root_uri: str

    def get_pipeline(self):
        from rastervision.new_version.pipeline.pipeline import Pipeline
        return Pipeline
