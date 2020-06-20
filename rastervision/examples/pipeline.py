from rastervision.pipeline.pipeline_config import PipelineConfig
from rastervision.aws_batch.aws_batch_runner import AWS_BATCH

def get_config(runner):
    root_uri = ('s3://raster-vision-lf-dev/examples/test-output/pipeline'
                if runner == AWS_BATCH else
                '/opt/data/examples/test-output/pipeline')
    return PipelineConfig(root_uri=root_uri)
