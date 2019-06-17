import keras
from rastervision.protos.keras_classification.optimizer_pb2 import Optimizer


def build(optimizer_options):
    if optimizer_options.type == Optimizer.Type.Value('ADAM'):
        optimizer = keras.optimizers.Adam(lr=optimizer_options.init_lr)
    elif optimizer_options.type == Optimizer.Type.Value('SGD'):
        optimizer = keras.optimizers.SGD(lr=optimizer_options.init_lr)
    else:
        raise ValueError((Optimizer.Type.Name(optimizer_options.type) +
                          ' is not a valid optimizer type'))

    return optimizer
