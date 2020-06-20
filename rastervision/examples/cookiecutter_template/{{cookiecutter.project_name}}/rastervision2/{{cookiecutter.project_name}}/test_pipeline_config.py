from rastervision2.pipeline.pipeline_config import PipelineConfig
from rastervision2.pipeline.config import register_config


@register_config('test_pipeline_config')
class TestPipelineConfig(PipelineConfig):
    message: str = 'hello'

    def build(self, tmp_dir):
        from rastervision2.{{cookiecutter.project_name}}.test_pipeline import TestPipeline
        return TestPipeline(self, tmp_dir)
