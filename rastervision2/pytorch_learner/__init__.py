# flake8: noqa

import rastervision2.pipeline
from rastervision2.pytorch_learner.learner_config import *
from rastervision2.pytorch_learner.learner import *
from rastervision2.pytorch_learner.learner_pipeline_config import *
from rastervision2.pytorch_learner.learner_pipeline import *
from rastervision2.pytorch_learner.classification_learner_config import *
from rastervision2.pytorch_learner.classification_learner import *
from rastervision2.pytorch_learner.regression_learner_config import *
from rastervision2.pytorch_learner.regression_learner import *
from rastervision2.pytorch_learner.semantic_segmentation_learner_config import *
from rastervision2.pytorch_learner.semantic_segmentation_learner import *
from rastervision2.pytorch_learner.object_detection_learner_config import *
from rastervision2.pytorch_learner.object_detection_learner import *


def register_plugin(registry):
    registry.set_plugin_version('rastervision2.pytorch_learner', 1)
