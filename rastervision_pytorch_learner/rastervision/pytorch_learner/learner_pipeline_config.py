from rastervision.pipeline.config import register_config
from rastervision.pipeline.pipeline_config import PipelineConfig
from rastervision.pytorch_learner import LearnerConfig


@register_config('learner_pipeline')
class LearnerPipelineConfig(PipelineConfig):
    """Configure a :class:`.LearnerPipeline`."""

    learner: LearnerConfig

    def update(self):
        super().update()

        if self.learner.output_uri is None:
            self.learner.output_uri = self.root_uri

        self.learner.update()

    def build(self, tmp_dir):
        from rastervision.pytorch_learner.learner_pipeline import LearnerPipeline
        return LearnerPipeline(self, tmp_dir)
