"""
Functions for training a model given a RunOptions object.
"""
from os.path import join, isfile

import numpy as np
from keras.callbacks import (ModelCheckpoint, CSVLogger,
                             ReduceLROnPlateau, LambdaCallback,
                             LearningRateScheduler)

from .data.generators import make_input_output_generators
from .data.settings import get_dataset_path, results_path
from .models.conv_logistic import make_conv_logistic
from .models.fcn_vgg import make_fcn_vgg
from .models.fcn_resnet import make_fcn_resnet
from .models.unet import make_unet

np.random.seed(1337)

CONV_LOGISTIC = 'conv_logistic'
FCN_VGG = 'fcn_vgg'
FCN_RESNET = 'fcn_resnet'
UNET = 'unet'


def make_model(options):
    """ A factory for generating models from options """
    model = None
    model_type = options.model_type
    if model_type == CONV_LOGISTIC:
        model = make_conv_logistic(options.input_shape, options.nb_labels,
                                   options.kernel_size)
    elif model_type == FCN_VGG:
        model = make_fcn_vgg(options.input_shape, options.nb_labels)
    elif model_type == FCN_RESNET:
        model = make_fcn_resnet(options.input_shape, options.nb_labels,
                                options.drop_prob, options.is_big_model)
    elif model_type == UNET:
        model = make_unet(options.input_shape, options.nb_labels)
    else:
        raise ValueError('{} is not a valid model_type'.format(model_type))

    return model


def train_model(model, sync_results, options):
    print(model.summary())
    path = get_dataset_path(options.dataset)
    train_generator, validation_generator = \
        make_input_output_generators(
            path, options.batch_size, options.include_depth)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
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

    model_checkpoint = ModelCheckpoint(
        filepath=join(run_path, 'model.h5'), period=1, save_weights_only=True)

    best_model_checkpoint = ModelCheckpoint(
        filepath=join(run_path, 'best_model.h5'), save_best_only=True,
        save_weights_only=True)
    logger = CSVLogger(log_path, append=True)
    callbacks = [model_checkpoint, best_model_checkpoint, logger]

    if options.patience:
        reduce_lr = ReduceLROnPlateau(
            verbose=1, epsilon=0.001, patience=options.patience)
        callbacks.append(reduce_lr)

    if options.lr_schedule:
        def get_lr(epoch):
            for epoch_thresh, lr in options.lr_schedule:
                if epoch >= epoch_thresh:
                    curr_lr = lr
                else:
                    break
            print(curr_lr)
            return curr_lr
        lr_scheduler = LearningRateScheduler(get_lr)
        callbacks.append(lr_scheduler)

    sync_results_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: sync_results())
    callbacks.append(sync_results_callback)

    model.fit_generator(
        train_generator,
        initial_epoch=initial_epoch,
        samples_per_epoch=options.samples_per_epoch,
        nb_epoch=options.nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=options.nb_val_samples,
        callbacks=callbacks)
