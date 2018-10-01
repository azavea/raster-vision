import os

from rastervision.backend.keras_classification.models.resnet50 import ResNet50
from rastervision.protos.keras_classification.model_pb2 import Model


def build_from_path(model_path):
    import keras

    return keras.models.load_model(model_path)


def build(model_options, pretrained_model_path):
    if os.path.isfile(model_options.model_path):
        return build_from_path(model_options.model_path)

    nb_channels = 3
    input_shape = (model_options.input_size, model_options.input_size,
                   nb_channels)
    activation = 'softmax'

    if model_options.type == Model.Type.Value('RESNET50'):
        model = ResNet50(
            include_top=True,
            weights=pretrained_model_path,
            load_weights_by_name=model_options.load_weights_by_name,
            input_shape=input_shape,
            classes=model_options.nb_classes,
            activation=activation)
    else:
        raise ValueError(
            Model.Type.Name(model_options.type) + ' is not a valid model_type')

    return model
