import os

import keras

from keras_classification.models.resnet50 import ResNet50
from keras_classification.protos.model_pb2 import Model


def build_from_path(model_path):
    return keras.models.load_model(model_path)


def build(model_options):
    if os.path.isfile(model_options.model_path):
        return build_from_path(model_options.model_path)

    nb_channels = 3
    input_shape = (model_options.input_size,
                   model_options.input_size,
                   nb_channels)
    activation = 'softmax'
    weights = 'imagenet'

    if model_options.type == Model.Type.Value('RESNET50'):
        model = ResNet50(
            include_top=True, weights=weights,
            input_shape=input_shape,
            classes=model_options.nb_classes,
            activation=activation)
    else:
        raise ValueError(
            Model.Type.Name(model_options.type) + ' is not a valid model_type')

    return model
