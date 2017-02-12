"""
Functions for training a model given a RunOptions object.
"""
import numpy as np
from os.path import join

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from process_data import make_input_output_generators, results_path

np.random.seed(1337)


def make_model(options):
    """ A factory for generating models from options """
    model = None
    model_type = options.model_type
    if model_type == 'conv_logistic':
        from conv_logistic import make_conv_logistic
        model = make_conv_logistic(options.input_shape, options.nb_labels)
    elif model_type == 'fcn_vgg':
        from fcn_vgg import make_fcn_vgg
        model = make_fcn_vgg(options.input_shape, options.nb_labels)

    return model


def train_model(model, options):
    print(model.summary())

    train_generator, validation_generator = \
        make_input_output_generators(options.batch_size)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    run_path = join(results_path, options.run_name)

    checkpoint = ModelCheckpoint(filepath=join(run_path, 'model.h5'),
                                 verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(patience=options.patience, verbose=1)
    logger = CSVLogger(join(run_path, 'log.txt'))
    callbacks = [checkpoint, early_stopping, logger]

    model.fit_generator(
        train_generator,
        samples_per_epoch=options.samples_per_epoch,
        nb_epoch=options.nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=options.nb_val_samples,
        callbacks=callbacks)
