"""
Functions for training a model given a RunOptions object.
"""
import numpy as np
from os.path import join, isfile

from keras.callbacks import (ModelCheckpoint, CSVLogger,
                             ReduceLROnPlateau)
from keras.optimizers import Adam

from .data.generators import make_input_output_generators
from .data.preprocess import get_dataset_path, results_path

np.random.seed(1337)


def make_model(options):
    """ A factory for generating models from options """
    model = None
    model_type = options.model_type
    if model_type == 'conv_logistic':
        from .models.conv_logistic import make_conv_logistic
        model = make_conv_logistic(options.input_shape, options.nb_labels,
                                   options.kernel_size)
    elif model_type == 'fcn_vgg':
        from .models.fcn_vgg import make_fcn_vgg
        model = make_fcn_vgg(options.input_shape, options.nb_labels)
    elif model_type == 'fcn_vgg_skip':
        from .models.fcn_vgg_skip import make_fcn_vgg_skip
        model = make_fcn_vgg_skip(options.input_shape, options.nb_labels)
    elif model_type == 'fcn_resnet':
        from .models.fcn_resnet import make_fcn_resnet
        model = make_fcn_resnet(options.input_shape, options.nb_labels)

    return model


def train_model(model, options):
    print(model.summary())
    path = get_dataset_path(options.dataset)
    train_generator, validation_generator = \
        make_input_output_generators(
            path, options.batch_size, options.include_depth)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=options.lr),
        metrics=['accuracy'])

    run_path = join(results_path, options.run_name)
    log_path = join(run_path, 'log.txt')

    initial_epoch = 0
    if isfile(log_path):
        with open(log_path) as log_file:
            line_ind = 0
            for line_ind, _ in enumerate(log_file):
                pass
            initial_epoch = line_ind

    checkpoint = ModelCheckpoint(filepath=join(run_path, 'model.h5'),
                                 verbose=1, save_best_only=True)
    logger = CSVLogger(log_path, append=True)
    reduce_lr = ReduceLROnPlateau(
        verbose=1, epsilon=0.001, patience=options.patience)

    callbacks = [checkpoint, logger, reduce_lr]

    model.fit_generator(
        train_generator,
        initial_epoch=initial_epoch,
        samples_per_epoch=options.samples_per_epoch,
        nb_epoch=options.nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=options.nb_val_samples,
        callbacks=callbacks)
