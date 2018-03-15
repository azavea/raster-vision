from rv2.ml_backends.tf_object_detection_api import TFObjectDetectionAPI
from rv2.ml_tasks.object_detection import ObjectDetection
from rv2.protos.machine_learning_pb2 import MachineLearning


def build(config):
    tf_object_detection_api_val = \
        MachineLearning.Backend.Value('TF_OBJECT_DETECTION_API')
    object_detection_val = \
        MachineLearning.Task.Value('OBJECT_DETECTION')

    if config.backend == tf_object_detection_api_val:
        backend = TFObjectDetectionAPI()

    if config.task == object_detection_val:
        task = ObjectDetection(backend)

    return task
