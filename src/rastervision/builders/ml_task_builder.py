from rastervision.ml_backends.tf_object_detection_api import (
    TFObjectDetectionAPI)
from rastervision.ml_tasks.object_detection import ObjectDetection
from rastervision.ml_backends.keras_classification import KerasClassification
from rastervision.ml_backends.tf_deeplab import TFDeeplab
from rastervision.ml_tasks.object_detection import ObjectDetection
from rastervision.ml_tasks.classification import Classification
from rastervision.ml_tasks.semantic_segmentation import SemanticSegmentation
from rastervision.protos.machine_learning_pb2 import MachineLearning
from rastervision.core.class_map import ClassItem, ClassMap


def build(config):
    class_items = []
    for item_config in config.class_items:
        item = ClassItem(item_config.id, item_config.name, item_config.color)
        class_items.append(item)
    class_map = ClassMap(class_items)

    tf_object_detection_api_val = \
        MachineLearning.Backend.Value('TF_OBJECT_DETECTION_API')
    object_detection_val = \
        MachineLearning.Task.Value('OBJECT_DETECTION')

    tf_deeplab_val = \
        MachineLearning.Backend.Value('TF_DEEPLAB')
    semantic_segmentation_val = \
        MachineLearning.Task.Value('SEMANTIC_SEGMENTATION')

    keras_classification_val = \
        MachineLearning.Backend.Value('KERAS_CLASSIFICATION')
    classification_val = \
        MachineLearning.Task.Value('CLASSIFICATION')

    backend_map = {
        tf_object_detection_api_val: TFObjectDetectionAPI,
        tf_deeplab_val: TFDeeplab,
        keras_classification_val: KerasClassification
    }

    task_map = {
        object_detection_val: ObjectDetection,
        semantic_segmentation_val: SemanticSegmentation,
        classification_val: Classification
    }

    # XXX backend_map and task_map may need to become a cross-product
    backend = (backend_map[config.backend])()
    task = (task_map[config.task])(backend, class_map)

    return task
