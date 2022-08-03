# flake8: noqa


def register_plugin(registry):
    registry.set_plugin_version('rastervision.pytorch_learner', 4)
    registry.set_plugin_aliases('rastervision.pytorch_learner',
                                ['rastervision2.pytorch_learner'])


import rastervision.pipeline
from rastervision.pytorch_learner.learner_config import *
from rastervision.pytorch_learner.learner import *
from rastervision.pytorch_learner.learner_pipeline_config import *
from rastervision.pytorch_learner.learner_pipeline import *
from rastervision.pytorch_learner.classification_learner_config import *
from rastervision.pytorch_learner.classification_learner import *
from rastervision.pytorch_learner.regression_learner_config import *
from rastervision.pytorch_learner.regression_learner import *
from rastervision.pytorch_learner.semantic_segmentation_learner_config import *
from rastervision.pytorch_learner.semantic_segmentation_learner import *
from rastervision.pytorch_learner.object_detection_learner_config import *
from rastervision.pytorch_learner.object_detection_learner import *
from rastervision.pytorch_learner.dataset import *
