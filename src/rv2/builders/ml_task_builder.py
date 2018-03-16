from rv2.ml_backends.tf_object_detection_api import TFObjectDetectionAPI
from rv2.ml_tasks.object_detection import ObjectDetection
from rv2.protos.machine_learning_pb2 import MachineLearning
from rv2.core.label_map import LabelItem, LabelMap


def build(config):
    label_items = []
    for label_item_config in config.label_items:
        item = LabelItem(label_item_config.id, label_item_config.name)
        label_items.append(item)
    label_map = LabelMap(label_items)

    tf_object_detection_api_val = \
        MachineLearning.Backend.Value('TF_OBJECT_DETECTION_API')
    object_detection_val = \
        MachineLearning.Task.Value('OBJECT_DETECTION')

    if config.backend == tf_object_detection_api_val:
        backend = TFObjectDetectionAPI()

    if config.task == object_detection_val:
        task = ObjectDetection(backend, label_map)

    return task
