from os.path import isfile, join
from subprocess import call

from .conv_logistic import make_conv_logistic, CONV_LOGISTIC
from .fcn_resnet import make_fcn_resnet, FCN_RESNET
from .dual_fcn_resnet import make_dual_fcn_resnet, DUAL_FCN_RESNET
from .unet import make_unet, UNET
from .fc_densenet import make_fc_densenet, FC_DENSENET
from .ensemble import (
    ConcatEnsemble, AvgEnsemble, CONCAT_ENSEMBLE, AVG_ENSEMBLE)
from ..data.settings import datasets_path, results_path, s3_bucket_name


def s3_download(run_name, file_name):
    s3_run_path = 's3://{}/results/{}'.format(
        s3_bucket_name, run_name)
    s3_file_path = join(s3_run_path, file_name)

    run_path = join(results_path, run_name)
    print(s3_file_path)
    print(run_path)
    call(['aws', 's3', 'cp', s3_file_path, run_path + '/'])


def load_ensemble_models(options):
    from ..options import load_options
    from ..data.factory import get_data_generator

    models = []
    active_input_inds_list = []

    for run_name in options.ensemble_run_names:
        s3_download(run_name, 'options.json')
        s3_download(run_name, 'best_model.h5')

        run_path = join(results_path, run_name)
        options_path = join(run_path, 'options.json')
        options = load_options(options_path)
        generator = get_data_generator(options, datasets_path)
        active_input_inds = generator.active_input_inds
        model = load_model(
            run_path, options, generator, use_best=True)

        models.append(model)
        active_input_inds_list.append(active_input_inds)

    return models, active_input_inds_list


def make_model(options, generator):
    """Make a new model."""
    model_type = options.model_type
    input_shape = (options.target_size[0], options.target_size[1],
                   len(options.active_input_inds))
    nb_labels = generator.dataset.nb_labels

    if model_type == CONV_LOGISTIC:
        model = make_conv_logistic(input_shape, nb_labels,
                                   options.kernel_size)
    elif model_type == FCN_RESNET:
        model = make_fcn_resnet(
            input_shape, nb_labels, options.use_pretraining,
            options.freeze_base)
    elif model_type == DUAL_FCN_RESNET:
        model = make_dual_fcn_resnet(
            input_shape, options.dual_active_input_inds,
            nb_labels, options.use_pretraining, options.freeze_base)
    elif model_type == UNET:
        model = make_unet(input_shape, nb_labels)
    elif model_type == FC_DENSENET:
        model = make_fc_densenet(
            input_shape, nb_labels, drop_prob=options.drop_prob,
            weight_decay=options.weight_decay,
            down_blocks=options.down_blocks,
            up_blocks=options.up_blocks)
    elif model_type in [CONCAT_ENSEMBLE, AVG_ENSEMBLE]:
        models, active_input_inds_list = load_ensemble_models(options)
        if model_type == CONCAT_ENSEMBLE:
            model = ConcatEnsemble(
                models, active_input_inds_list, input_shape, nb_labels)
        elif model_type == AVG_ENSEMBLE:
            model = AvgEnsemble(models, active_input_inds_list)
    else:
        raise ValueError('{} is not a valid model_type'.format(model_type))

    return model


def load_model(run_path, options, generator, use_best=True):
    """Load an existing model."""
    # Load the model by weights. This permits loading weights from a saved
    # model into a model with a different architecture assuming the named
    # layers have compatible dimensions.
    model = make_model(options, generator)
    file_name = 'best_model.h5' if use_best else 'model.h5'
    model_path = join(run_path, file_name)
    # TODO raise exception if model_path doesn't exist
    model.load_weights(model_path, by_name=True)
    return model


def get_model(run_path, options, generator, use_best=True):
    """Get a model by loading if it exists or making a new one."""
    model_path = join(run_path, 'model.h5')

    # Load the model if it's saved, or create a new one.
    if isfile(model_path):
        model = load_model(run_path, options, generator, use_best)
        print('Continuing training from saved model.')
    else:
        model = make_model(options, generator)
        print('Creating new model.')

    return model
