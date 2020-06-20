from rastervision.pipeline.pipeline import Pipeline
from rastervision.pytorch_learner import LearnerConfig


class LearnerPipeline(Pipeline):
    """Simple Pipeline that is a wrapper around Learner.main()

    This supports the ability to use the pytorch_learner package to train models using
    the RV pipeline package and its runner functionality without the rest of RV.
    """
    commands = ['train']
    gpu_commands = ['train']

    def train(self):
        learner_cfg: LearnerConfig = self.config.learner
        learner = learner_cfg.build(learner_cfg, self.tmp_dir)
        learner.main()
