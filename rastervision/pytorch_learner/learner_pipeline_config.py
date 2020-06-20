from rastervision2.pipeline.config import register_config
from rastervision2.pipeline.pipeline_config import PipelineConfig
from rastervision2.pytorch_learner import LearnerConfig


@register_config('learner_pipeline')
class LearnerPipelineConfig(PipelineConfig):
    learner: LearnerConfig

    def update(self):
        super().update()

        if self.learner.output_uri is None:
            self.learner.output_uri = self.root_uri

        self.learner.update()

    def build(self, tmp_dir):
        from rastervision2.pytorch_learner.learner_pipeline import LearnerPipeline
        return LearnerPipeline(self, tmp_dir)
