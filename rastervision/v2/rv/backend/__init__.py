# flake8: noqa

from rastervision.v2.backend.backend import *
from rastervision.v2.backend.backend_config import *

try:
    import tensorflow
    tf_available = True
except ImportError as e:
    tf_available = False

try:
    import torch
    pytorch_available = True
except ImportError as e:
    pytorch_available = False

if tf_available:
    from rastervision.v2.backend.tf_object_detection import TFObjectDetection
    from rastervision.v2.backend.tf_object_detection_config import (
        TFObjectDetectionConfig, TFObjectDetectionConfigBuilder)
    from rastervision.v2.backend.keras_classification import KerasClassification
    from rastervision.v2.backend.keras_classification_config import (
        KerasClassificationConfig, KerasClassificationConfigBuilder)
    from rastervision.v2.backend.tf_deeplab import TFDeeplab
    from rastervision.v2.backend.tf_deeplab_config import (TFDeeplabConfig,
                                                        TFDeeplabConfigBuilder)

if pytorch_available:
    from rastervision.v2.backend.pytorch_chip_classification import (
        PyTorchChipClassification)
    from rastervision.v2.backend.pytorch_chip_classification_config import (
        PyTorchChipClassificationConfig,
        PyTorchChipClassificationConfigBuilder)
    from rastervision.v2.backend.pytorch_semantic_segmentation import (
        PyTorchSemanticSegmentation)
    from rastervision.v2.backend.pytorch_semantic_segmentation_config import (
        PyTorchSemanticSegmentationConfig,
        PyTorchSemanticSegmentationConfigBuilder)
    from rastervision.v2.backend.pytorch_object_detection_config import (
        PyTorchObjectDetectionConfig, PyTorchObjectDetectionConfigBuilder)
