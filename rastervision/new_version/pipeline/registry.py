from rastervision.new_version.pipeline.inprocess_runner import InProcessRunner, INPROCESS
from rastervision.new_version.pipeline.aws_batch_runner import AWSBatchRunner, AWS_BATCH
from rastervision.new_version.pipeline.pipeline import Pipeline, BASE_PIPELINE


class Registry():
    def __init__(self):
        self.registry = {
            'runners': {
                INPROCESS: InProcessRunner,
                AWS_BATCH: AWSBatchRunner
            },
            'pipelines': {
                BASE_PIPELINE: Pipeline
            },
        }

    def get_runner(self, runner_type):
        return self.registry['runners'][runner_type]

    def get_pipeline(self, pipeline_type):
        return self.registry['pipelines'][pipeline_type]

    def add_pipeline(self, pipeline_type, pipeline):
        self.registry['pipelines'][pipeline_type] = pipeline


registry = Registry()
