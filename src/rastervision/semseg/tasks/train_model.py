from os.path import join, isfile
import math

from keras.callbacks import (Callback, ModelCheckpoint, CSVLogger,
                             ReduceLROnPlateau, LambdaCallback,
                             LearningRateScheduler)
from keras.optimizers import Adam, RMSprop

from rastervision.common.utils import _makedirs
from rastervision.common.settings import TRAIN, VALIDATION


ADAM = 'adam'
RMS_PROP = 'rms_prop'

TRAIN_MODEL = 'train_model'


class DeltaModelCheckpoint(Callback):
    def __init__(self, file_path, acc_delta=0.01):
        self.file_path = file_path
        self.acc_delta = acc_delta
        self.last_acc = -1
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        acc = logs['acc']
        if acc - self.last_acc > self.acc_delta:
            self.last_acc = acc
            self.model.save_weights(self.file_path.format(epoch=epoch))


def make_callbacks(run_path, sync_results, options, log_path):
    model_checkpoint = ModelCheckpoint(
        filepath=join(run_path, 'model.h5'), period=1, save_weights_only=True)

    best_model_checkpoint = ModelCheckpoint(
        filepath=join(run_path, 'best_model.h5'), save_best_only=True,
        save_weights_only=True)
    logger = CSVLogger(log_path, append=True)
    callbacks = [model_checkpoint, best_model_checkpoint, logger]

    if options.delta_model_checkpoint is not None:
        exp_path = join(run_path, 'delta_model_checkpoints')
        _makedirs(exp_path)
        callback = DeltaModelCheckpoint(
            join(exp_path, 'model_{epoch:0>4}.h5'),
            acc_delta=options.delta_model_checkpoint)
        callbacks.append(callback)

    if options.patience:
        callback = ReduceLROnPlateau(
            verbose=1, epsilon=0.001, patience=options.patience)
        callbacks.append(callback)

    if options.lr_schedule:
        def get_lr(epoch):
            for epoch_thresh, lr in options.lr_schedule:
                if epoch >= epoch_thresh:
                    curr_lr = lr
                else:
                    break
            return curr_lr
        callback = LearningRateScheduler(get_lr)
        callbacks.append(callback)

    callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: sync_results())
    callbacks.append(callback)

    return callbacks


def get_initial_epoch(log_path):
    """Get initial_epoch from the last line in the log csv file."""
    initial_epoch = 0
    if isfile(log_path):
        with open(log_path) as log_file:
            line_ind = 0
            for line_ind, _ in enumerate(log_file):
                pass
            initial_epoch = line_ind

    return initial_epoch


def get_lr(epoch, lr_schedule):
    for epoch_thresh, lr in lr_schedule:
        if epoch >= epoch_thresh:
            curr_lr = lr
        else:
            break
    return curr_lr


def train_model(run_path, model, sync_results, options, generator):
    """Train a model according to options using generator.

    This saves results after each epoch and attempts to resume training
    from the last saved point.

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
        shuffle=True, augment=True, normalize=True, only_xy=True)
    validation_gen = generator.make_split_generator(
        VALIDATION, target_size=options.target_size,
        batch_size=options.batch_size,
        shuffle=True, augment=True, normalize=True, only_xy=True)

    if options.optimizer == ADAM:
        optimizer = Adam(lr=options.init_lr)
    elif options.optimizer == RMS_PROP:
        optimizer = RMSprop(lr=options.init_lr)

    model.compile(
        optimizer, 'categorical_crossentropy', metrics=['accuracy'])

    log_path = join(run_path, 'log.txt')
    initial_epoch = get_initial_epoch(log_path)

    callbacks = make_callbacks(run_path, sync_results, options, log_path)

    model.fit_generator(
        train_gen,
        initial_epoch=initial_epoch,
        steps_per_epoch=options.steps_per_epoch,
        epochs=options.epochs,
        validation_data=validation_gen,
        validation_steps=options.validation_steps,
        callbacks=callbacks)
