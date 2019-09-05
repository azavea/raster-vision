# flake8: noqa

from rastervision.backend.backend import *
from rastervision.backend.backend_config import *

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
    from rastervision.backend.tf_object_detection import TFObjectDetection
    from rastervision.backend.tf_object_detection_config import (
        TFObjectDetectionConfig, TFObjectDetectionConfigBuilder)
    from rastervision.backend.keras_classification import KerasClassification
    from rastervision.backend.keras_classification_config import (
        KerasClassificationConfig, KerasClassificationConfigBuilder)
    from rastervision.backend.tf_deeplab import TFDeeplab
    from rastervision.backend.tf_deeplab_config import (TFDeeplabConfig,
                                                        TFDeeplabConfigBuilder)

if pytorch_available:
    from rastervision.backend.pytorch_chip_classification import (
        PyTorchChipClassification)
    from rastervision.backend.pytorch_chip_classification_config import (
        PyTorchChipClassificationConfig,
        PyTorchChipClassificationConfigBuilder)
    from rastervision.backend.pytorch_semantic_segmentation import (
        PyTorchSemanticSegmentation)
    from rastervision.backend.pytorch_semantic_segmentation_config import (
        PyTorchSemanticSegmentationConfig,
        PyTorchSemanticSegmentationConfigBuilder)
