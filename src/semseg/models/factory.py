from os.path import isfile, join

from .conv_logistic import make_conv_logistic, CONV_LOGISTIC
from .fcn_vgg import make_fcn_vgg, FCN_VGG
from .fcn_resnet import make_fcn_resnet, FCN_RESNET
from .unet import make_unet, UNET
from .fc_densenet import make_fc_densenet, FC_DENSENET


def make_model(options, dataset):
    """ A factory for generating models from options """
    model_type = options.model_type
    input_shape = (
        options.tile_size[0], options.tile_size[1], dataset.nb_channels)
    nb_labels = dataset.nb_labels

    if model_type == CONV_LOGISTIC:
        model = make_conv_logistic(input_shape, nb_labels,
                                   options.kernel_size)
    elif model_type == FCN_VGG:
        model = make_fcn_vgg(input_shape, nb_labels)
    elif model_type == FCN_RESNET:
        model = make_fcn_resnet(input_shape, nb_labels,
                                options.drop_prob, options.is_big_model)
    elif model_type == UNET:
        model = make_unet(input_shape, nb_labels)
    elif model_type == FC_DENSENET:
        model = make_fc_densenet(
            input_shape, nb_labels, drop_prob=options.drop_prob,
            weight_decay=options.weight_decay,
            down_blocks=options.down_blocks,
            up_blocks=options.up_blocks)
    else:
        raise ValueError('{} is not a valid model_type'.format(model_type))

    return model


def load_model(run_path, options, dataset, use_best=True):
    # Load the model by weights. This permits loading weights from a saved
    # model into a model with a different architecture assuming the named
    # layers have compatible dimensions.
    model = make_model(options, dataset)
    file_name = 'best_model.h5' if use_best else 'model.h5'
    model_path = join(run_path, file_name)
    # TODO raise exception if model_path doesn't exist
    model.load_weights(model_path, by_name=True)
    return model


def get_model(run_path, options, dataset, use_best=True):
    model_path = join(run_path, 'model.h5')

    # Load the model if it's saved, or create a new one.
    if isfile(model_path):
        model = load_model(run_path, options, dataset, use_best)
        print('Continuing training from saved model.')
    else:
        model = make_model(options, dataset)
        print('Creating new model.')

    return model
