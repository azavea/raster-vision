# flake8: noqa


def register_plugin(registry):
    registry.set_plugin_version('rastervision.pytorch_backend', 1)


import rastervision.pipeline
from rastervision.pytorch_backend.pytorch_chip_classification_config import *
from rastervision.pytorch_backend.pytorch_chip_classification import *
from rastervision.pytorch_backend.pytorch_semantic_segmentation_config import *
from rastervision.pytorch_backend.pytorch_semantic_segmentation import *
from rastervision.pytorch_backend.pytorch_object_detection_config import *
from rastervision.pytorch_backend.pytorch_object_detection import *

__all__ = [
    PyTorchChipClassification.__name__,
    PyTorchChipClassificationConfig.__name__,
    PyTorchSemanticSegmentation.__name__,
    PyTorchSemanticSegmentationConfig.__name__,
    PyTorchObjectDetection.__name__,
    PyTorchObjectDetectionConfig.__name__,
]
