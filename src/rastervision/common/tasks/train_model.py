from os.path import join, isfile

from keras.callbacks import (Callback, ModelCheckpoint, CSVLogger,
                             ReduceLROnPlateau, LambdaCallback,
                             LearningRateScheduler)

from keras.optimizers import Adam, RMSprop, SGD, TFOptimizer
from rastervision.common.optimizers.yellowfin import YFOptimizer
from rastervision.common.tasks.cyclic_lr import CyclicLR
from rastervision.common.utils import _makedirs
from rastervision.common.settings import TRAIN, VALIDATION


ADAM = 'adam'
RMS_PROP = 'rms_prop'
YELLOWFIN = 'yellowfin'

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


class TrainModel():
    def __init__(self, run_path, sync_results, options,
                 generator, model):
        self.run_path = run_path
        self.sync_results = sync_results
        self.options = options
        self.generator = generator
        self.model = model

        self.log_path = join(self.run_path, 'log.txt')

        # Should be overridden
        self.metrics = None
        self.loss_function = None

    def make_callbacks(self):
        model_checkpoint = ModelCheckpoint(
            filepath=join(self.run_path, 'model.h5'), period=1,
            save_weights_only=True)

        best_model_checkpoint = ModelCheckpoint(
            filepath=join(self.run_path, 'best_model.h5'), save_best_only=True,
            save_weights_only=True)
        logger = CSVLogger(self.log_path, append=True)
        callbacks = [model_checkpoint, best_model_checkpoint, logger]

        # TODO hasattr
        if self.options.delta_model_checkpoint is not None:
            exp_path = join(self.run_path, 'delta_model_checkpoints')
            _makedirs(exp_path)
            callback = DeltaModelCheckpoint(
                join(exp_path, 'model_{epoch:0>4}.h5'),
                acc_delta=self.options.delta_model_checkpoint)
            callbacks.append(callback)

        if self.options.patience:
            callback = ReduceLROnPlateau(
                verbose=1, epsilon=0.001, patience=self.options.patience)
            callbacks.append(callback)

        if self.options.lr_schedule:
            def get_lr(epoch):
                for epoch_thresh, lr in self.options.lr_schedule:
                    if epoch >= epoch_thresh:
                        curr_lr = lr
                    else:
                        break
                return curr_lr
            callback = LearningRateScheduler(get_lr)
            callbacks.append(callback)

        if self.options.lr_epoch_decay:
            def get_lr(epoch):
                decay_factor = 1 / (1.0 + self.options.lr_epoch_decay * epoch)
                return self.options.init_lr * decay_factor
            callback = LearningRateScheduler(get_lr)
            callbacks.append(callback)

        if self.options.cyclic_lr is not None:
            callback = CyclicLR(base_lr=self.options.base_lr,
                                max_lr=self.options.max_lr,
                                step_size=self.options.step_size,
                                mode=self.options.cycle_mode)
            callbacks.append(callback)

        callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: self.sync_results())
        callbacks.append(callback)

        return callbacks

    def get_initial_epoch(self):
        """Get initial_epoch from the last line in the log csv file."""
        initial_epoch = 0
        if isfile(self.log_path):
            with open(self.log_path) as log_file:
                line_ind = 0
                for line_ind, _ in enumerate(log_file):
                    pass
                initial_epoch = line_ind

        return initial_epoch

    def train_model(self):
        """Train a model according to options using generator.

        This saves results after each epoch and attempts to resume training
        from the last saved point.

        # Arguments
            run_path: the path to the files for a run
            model: a Keras model
            options: RunOptions object that specifies the run
            generator: a Generator object to generate the training and
                validation data
        """
        print(self.model.summary())

        train_gen = self.generator.make_split_generator(
            TRAIN, target_size=self.options.target_size,
            batch_size=self.options.batch_size,
            shuffle=True, augment_methods=self.options.augment_methods,
            normalize=True, only_xy=True)
        validation_gen = self.generator.make_split_generator(
            VALIDATION, target_size=self.options.target_size,
            batch_size=self.options.batch_size,
            shuffle=True, augment_methods=self.options.augment_methods,
            normalize=True, only_xy=True)

        if self.options.optimizer == ADAM:
            optimizer = Adam(lr=self.options.init_lr,
                             decay=self.options.lr_step_decay)
        elif self.options.optimizer == RMS_PROP:
            optimizer = RMSprop(lr=self.options.init_lr,
                                rho=self.options.rho,
                                epsilon=self.options.epsilon,
                                decay=self.options.lr_step_decay)
        elif self.options.optimizer == 'sgd':
            optimizer = SGD(
                lr=self.options.init_lr,
                momentum=self.options.momentum,
                nesterov=self.options.nesterov
            )
        elif self.options.optimizer == YELLOWFIN:
            optimizer = TFOptimizer(YFOptimizer(
                learning_rate=self.options.init_lr,
                momentum=self.options.momentum))

        self.model.compile(
            optimizer, self.loss_function, metrics=self.metrics)

        initial_epoch = self.get_initial_epoch()

        callbacks = self.make_callbacks()

        self.model.fit_generator(
            train_gen,
            initial_epoch=initial_epoch,
            steps_per_epoch=self.options.steps_per_epoch,
            epochs=self.options.epochs,
            validation_data=validation_gen,
            validation_steps=self.options.validation_steps,
            callbacks=callbacks)
