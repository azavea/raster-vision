from os.path import join

from rastervision.aws_batch.aws_batch_runner import AWS_BATCH
from rastervision.pytorch_learner.classification_learner_config import (
    ClassificationLearnerConfig, ClassificationDataConfig)
from rastervision.pytorch_learner.learner_config import (SolverConfig,
                                                         ModelConfig)
from rastervision.pytorch_learner.learner_pipeline_config import LearnerPipelineConfig


def get_config(runner, test=False):
    base_uri = ('s3://raster-vision-lf-dev/learner/classification'
                if runner == AWS_BATCH else '/opt/data/learner/classification')
    root_uri = join(base_uri, 'output')
    data_uri = join(base_uri, 'tiny-buildings.zip')

    model = ModelConfig(backbone='resnet50')
    solver = SolverConfig(lr=2e-4, num_epochs=3, batch_sz=8, one_cycle=True)
    data = ClassificationDataConfig(
        data_format='image_folder',
        uri=data_uri,
        img_sz=200,
        labels=['building', 'no_building'])
    learner = ClassificationLearnerConfig(
        model=model, solver=solver, data=data, test_mode=test)
    pipeline = LearnerPipelineConfig(root_uri=root_uri, learner=learner)
    return pipeline
