import os

from keras_classification.models.resnet50 import ResNet50
from keras_classification.protos.model_pb2 import Model


def build(model_options):
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

    if os.path.isfile(model_options.model_path):
        # Load the model by weights. This permits loading weights from a saved
        # model into a model with a different architecture assuming the named
        # layers have compatible dimensions.
        model.load_weights(model_options.model_path, by_name=True)

    return model
