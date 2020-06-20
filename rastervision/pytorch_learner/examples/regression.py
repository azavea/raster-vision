from os.path import join

from rastervision.aws_batch.aws_batch_runner import AWS_BATCH
from rastervision.pytorch_learner.regression.config import (
    RegressionLearnerConfig, RegressionDataConfig, RegressionModelConfig)
from rastervision.pytorch_learner.learner_config import (SolverConfig)
from rastervision.pytorch_learner.learner_pipeline_config import LearnerPipelineConfig


def get_config(runner, test=False):
    base_uri = ('s3://raster-vision-lf-dev/learner/regression'
                if runner == AWS_BATCH else '/opt/data/learner/regression')
    root_uri = join(base_uri, 'output')
    data_uri = join(base_uri, 'tiny-buildings.zip')

    model = RegressionModelConfig(backbone='resnet50')
    solver = SolverConfig(lr=1e-4, num_epochs=10, batch_sz=8, one_cycle=True)
    data = RegressionDataConfig(
        data_format='image_csv',
        uri=data_uri,
        img_sz=200,
        labels=['has_buildings'])
    learner = RegressionLearnerConfig(
        model=model, solver=solver, data=data, test_mode=test)

    pipeline = LearnerPipelineConfig(root_uri=root_uri, learner=learner)
    return pipeline
