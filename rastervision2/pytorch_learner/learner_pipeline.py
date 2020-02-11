from rastervision2.pipeline.pipeline import Pipeline

LEARNER_PIPELINE = 'learner_pipeline'


class LearnerPipeline(Pipeline):
    commands = ['train']
    gpu_commands = ['train']

    def train(self):
        learner_cfg = self.config.learner
        learner = learner_cfg.build(learner_cfg, self.tmp_dir)
        learner.main()
