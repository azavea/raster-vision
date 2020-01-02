from rastervision.v2.core.pipeline import Pipeline
from rastervision.v2.learner.learner_config import LearnerPipelineConfig

LEARNER_PIPELINE = 'learner_pipeline'


class LearnerPipeline(Pipeline):
    config_class = LearnerPipelineConfig
    commands = ['train']
    gpu_commands = ['train']

    def train(self):
        learner_cfg = self.config.learner
        learner_cls = learner_cfg.get_learner()
        learner = learner_cls(learner_cfg, self.tmp_dir)
        learner.main()
