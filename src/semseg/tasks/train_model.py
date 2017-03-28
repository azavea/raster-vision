from os.path import join, isfile

from keras.callbacks import (ModelCheckpoint, CSVLogger,
                             ReduceLROnPlateau, LambdaCallback,
                             LearningRateScheduler)
from keras.optimizers import Adam, RMSprop

from ..data.generators import TRAIN, VALIDATION

ADAM = 'adam'
RMS_PROP = 'rms_prop'

TRAIN_MODEL = 'train_model'


def train_model(run_path, model, sync_results, options, generator):
    """Train a model according to options using generator.

    This saves results after each epoch and attempts to resume training
    at the last saved point.

    # Arguments
        run_path: the path to the files for a run
        model: a Keras model
        options: RunOptions object that specifies the run
        generator: a Generator object to generate the training and validation
            data
    """
    print(model.summary())

    train_gen = generator.make_split_generator(
        TRAIN, target_size=options.target_size, batch_size=options.batch_size,
        shuffle=True, augment=True, normalize=True)
    validation_gen = generator.make_split_generator(
        VALIDATION, target_size=options.target_size, batch_size=options.batch_size,
        shuffle=True, augment=True, normalize=True)

    if options.optimizer == ADAM:
        optimizer = Adam(options.init_lr)
    elif options.optimizer == RMS_PROP:
        optimizer = RMSprop(options.init_lr)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

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
