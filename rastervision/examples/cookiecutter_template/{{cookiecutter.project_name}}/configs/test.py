from rastervision.{{cookiecutter.project_name}}.test_pipeline_config import (
    TestPipelineConfig)


def get_config(runner, root_uri='/opt/data/test-pipeline', message='hello world'):
    return TestPipelineConfig(root_uri=root_uri, message=message)
