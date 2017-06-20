from rastervision.common.models.factory import ModelFactory
from rastervision.common.models.resnet50 import ResNet50
from rastervision.common.models.wideresnet import WideResidualNetwork
from rastervision.semseg.models.fcn_wrn import make_fcn_wrn, FCN_WRN

BASELINE_RESNET = 'baseline_resnet'
WIDERESNET = 'wideresnet'

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
                include_top=False,
                weights=weights,
                input_shape=input_shape,
                classes=generator.dataset.nb_tags,
                activation='sigmoid')
        elif model_type == WIDERESNET:
            weights = 'cifar10' if options.use_pretraining else None
            model = WideResidualNetwork(
                nb_classes=generator.dataset.nb_tags,
                include_top=True,
                weights=weights,
                #classes=generator.dataset.nb_tags,
                input_shape=input_shape,
                activation='sigmoid')
        elif model_type == FCN_WRN:
            model = make_fcn_wrn(
                input_shape,
                generator.dataset.nb_tags,
                use_pretraining=options.use_pretraining,
                activation='sigmoid')
        else:
            raise ValueError('{} is not a valid model_type'.format(model_type))

        return model
