from rastervision.new_version.pipeline.pipeline_config import PipelineConfig


def get_config(runner, root_uri=None):
    return PipelineConfig(root_uri=root_uri)
