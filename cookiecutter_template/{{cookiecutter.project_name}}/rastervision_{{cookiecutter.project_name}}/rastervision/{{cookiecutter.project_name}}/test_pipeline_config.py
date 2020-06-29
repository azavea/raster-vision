from rastervision.pipeline.pipeline_config import PipelineConfig
from rastervision.pipeline.config import register_config


@register_config('{{cookiecutter.project_name}}.test_pipeline_config')
class TestPipelineConfig(PipelineConfig):
    message: str = 'hello'

    def build(self, tmp_dir):
        from rastervision.{{cookiecutter.project_name}}.test_pipeline import TestPipeline
        return TestPipeline(self, tmp_dir)
