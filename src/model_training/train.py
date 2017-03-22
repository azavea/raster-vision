"""
Functions for training a model given a RunOptions object.
"""
from os.path import join, isfile

from keras.callbacks import (ModelCheckpoint, CSVLogger,
                             ReduceLROnPlateau, LambdaCallback,
                             LearningRateScheduler)
from keras.optimizers import Adam, RMSprop

from .data.settings import results_path
from .data.datasets import TRAIN, VALIDATION
from .models.conv_logistic import make_conv_logistic
from .models.fcn_vgg import make_fcn_vgg
from .models.fcn_resnet import make_fcn_resnet
from .models.unet import make_unet
from .models.fc_densenet import make_fc_densenet

CONV_LOGISTIC = 'conv_logistic'
FCN_VGG = 'fcn_vgg'
FCN_RESNET = 'fcn_resnet'
UNET = 'unet'
FC_DENSENET = 'fc_densenet'

ADAM = 'adam'
RMS_PROP = 'rms_prop'

def make_model(options, dataset):
    """ A factory for generating models from options """
    model_type = options.model_type
    input_shape = dataset.input_shape
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


def train_model(model, sync_results, options, generator):
    print(model.summary())

    train_gen = generator.make_split_generator(
        TRAIN, batch_size=options.batch_size, shuffle=True, augment=True,
        normalize=True)
    validation_gen = generator.make_split_generator(
        VALIDATION, batch_size=options.batch_size, shuffle=True,
        augment=True, normalize=True)

    if options.optimizer == ADAM:
        optimizer = Adam(options.init_lr)
    elif options.optimizer == RMS_PROP:
        optimizer = RMSprop(options.init_lr)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
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
            return curr_lr
        lr_scheduler = LearningRateScheduler(get_lr)
        callbacks.append(lr_scheduler)

    sync_results_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: sync_results())
    callbacks.append(sync_results_callback)

    model.fit_generator(
        train_gen,
        initial_epoch=initial_epoch,
        samples_per_epoch=options.samples_per_epoch,
        nb_epoch=options.nb_epoch,
        validation_data=validation_gen,
        nb_val_samples=options.nb_val_samples,
        callbacks=callbacks)
