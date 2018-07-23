import keras

from keras_classification.protos.optimizer_pb2 import Optimizer


def build(optimizer_options):
    if optimizer_options.type == Optimizer.Type.Value('ADAM'):
        optimizer = keras.optimizers.Adam(lr=optimizer_options.init_lr)
    else:
        raise ValueError((Optimizer.Type.Name(optimizer_options.type) +
                          ' is not a valid optimizer type'))

    return optimizer
