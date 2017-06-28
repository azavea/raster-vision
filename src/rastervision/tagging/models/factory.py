from rastervision.common.models.factory import ModelFactory
from rastervision.common.models.resnet50 import ResNet50
from rastervision.common.models.densenet121 import DenseNet121
from rastervision.common.models.densenet169 import DenseNet169

BASELINE_RESNET = 'baseline_resnet'
DENSENET_121 = 'densenet121'
DENSENET_169 = 'densenet169'


class TaggingModelFactory(ModelFactory):
    def __init__(self):
        super().__init__()

    def make_model(self, options, generator):
        """Make a new model."""
        model_type = options.model_type
        nb_channels = len(options.active_input_inds)
        image_shape = generator.dataset.image_shape
        input_shape = (image_shape[0], image_shape[1], nb_channels)

        if model_type == BASELINE_RESNET:
            # A ResNet50 model with sigmoid activation and binary_crossentropy
            # as a loss function.
            weights = 'imagenet' if options.use_pretraining else None
            model = ResNet50(
                include_top=True, weights=weights,
                input_shape=input_shape,
                classes=len(generator.active_tags),
                activation='sigmoid')
        elif model_type == DENSENET_121:
            weights = 'imagenet' if options.use_pretraining else None
            model = DenseNet121(weights=weights,
                                input_shape=input_shape,
                                classes=len(generator.tag_store.active_tags),
                                activation='sigmoid')
        elif model_type == DENSENET_169:
            weights = 'imagenet' if options.use_pretraining else None
            model = DenseNet169(weights=weights,
                                input_shape=input_shape,
                                classes=len(generator.tag_store.active_tags),
                                activation='sigmoid')
        else:
            raise ValueError('{} is not a valid model_type'.format(model_type))

        if options.freeze_base:
            for layer in model.layers[:-1]:
                layer.trainable = False

        return model
