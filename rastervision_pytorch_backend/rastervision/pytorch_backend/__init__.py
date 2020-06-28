# flake8: noqa

import rastervision.pipeline
from rastervision.pytorch_backend.pytorch_chip_classification_config import *
from rastervision.pytorch_backend.pytorch_chip_classification import *
from rastervision.pytorch_backend.pytorch_semantic_segmentation_config import *
from rastervision.pytorch_backend.pytorch_semantic_segmentation import *
from rastervision.pytorch_backend.pytorch_object_detection_config import *
from rastervision.pytorch_backend.pytorch_object_detection import *


def register_plugin(registry):
    registry.set_plugin_version('rastervision.pytorch_backend', 0)
    registry.set_plugin_aliases('rastervision.pytorch_backend',
                                ['rastervision2.pytorch_backend'])
