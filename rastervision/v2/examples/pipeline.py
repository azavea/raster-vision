from rastervision.v2.core.pipeline_config import PipelineConfig

def get_config(runner):
    root_uri = (
        's3://raster-vision-lf-dev/examples/test-output/pipeline'
        if runner == 'aws_batch'
        else '/opt/data/examples/test-output/pipeline')
    return PipelineConfig(root_uri=root_uri)
