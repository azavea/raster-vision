from rv2.ml_backends.tf_object_detection_api import TFObjectDetectionAPI
from rv2.ml_tasks.object_detection import ObjectDetection
from rv2.ml_backends.keras_classification import KerasClassification
from rv2.ml_tasks.classification import Classification
from rv2.protos.machine_learning_pb2 import MachineLearning
from rv2.core.class_map import ClassItem, ClassMap


def build(config):
    class_items = []
    for item_config in config.class_items:
        item = ClassItem(item_config.id, item_config.name)
        class_items.append(item)
    class_map = ClassMap(class_items)

    tf_object_detection_api_val = \
        MachineLearning.Backend.Value('TF_OBJECT_DETECTION_API')
    object_detection_val = \
        MachineLearning.Task.Value('OBJECT_DETECTION')

    keras_classification_val = \
        MachineLearning.Backend.Value('KERAS_CLASSIFICATION')
    classification_val = \
        MachineLearning.Task.Value('CLASSIFICATION')

    if config.backend == tf_object_detection_api_val:
        backend = TFObjectDetectionAPI()
    elif config.backend == keras_classification_val:
        backend = KerasClassification()

    if config.task == object_detection_val:
        task = ObjectDetection(backend, class_map)
    elif config.task == classification_val:
        task = Classification(backend, class_map)

    return task
