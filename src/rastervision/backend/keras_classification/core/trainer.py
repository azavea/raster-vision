import os
from pathlib import Path
from subprocess import Popen
import logging

from rastervision.utils.misc import terminate_at_exit
from rastervision.backend.keras_classification.utils import make_dir
log = logging.getLogger(__name__)


def get_nb_images(image_dir):
    count = 0

    pathlist = Path(image_dir).glob('**/*.png')
    for path in pathlist:
        count += 1

    pathlist = Path(image_dir).glob('**/*.jpg')
    for path in pathlist:
        count += 1

    return count


class Trainer(object):
    def __init__(self, model, optimizer, options):
        self.model = model
        self.optimizer = optimizer
        self.options = options

        self.model_path = os.path.join(options.output_dir, 'model')
        make_dir(self.model_path, use_dirname=True)
        self.weights_path = os.path.join(options.output_dir,
                                         'model-weights.hdf5')
        make_dir(self.weights_path, use_dirname=True)
        self.log_path = os.path.join(options.output_dir, 'log.csv')
        make_dir(self.log_path, use_dirname=True)

        self.training_gen = self.make_data_generator(options.training_data_dir)
        self.validation_gen = self.make_data_generator(
            options.validation_data_dir, validation_mode=True)

        self.nb_training_samples = get_nb_images(options.training_data_dir)
        self.nb_validation_samples = get_nb_images(options.validation_data_dir)

        self.tf_logs_path = os.path.join(options.output_dir, 'logs')

    def make_callbacks(self, do_monitoring):
        import keras

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self.model_path,
            save_best_only=self.options.save_best,
            save_weights_only=False)

        weights_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self.weights_path,
            save_best_only=self.options.save_best,
            save_weights_only=True)

        csv_logger = keras.callbacks.CSVLogger(self.log_path, append=True)

        callbacks = [model_checkpoint, weights_checkpoint, csv_logger]

        if self.options.lr_schedule:
            lr_schedule = sorted(
                self.options.lr_schedule, key=lambda x: x.epoch, reverse=True)

            def schedule(curr_epoch):
                for lr_schedule_item in lr_schedule:
                    if curr_epoch >= lr_schedule_item.epoch:
                        if self.options.debug:
                            log.info('New lr: {}'.format(lr_schedule_item.lr))
                        return lr_schedule_item.lr

            lr_scheduler = keras.callbacks.LearningRateScheduler(schedule)

            callbacks.append(lr_scheduler)

        if do_monitoring:
            tensorboard = keras.callbacks.TensorBoard(
                log_dir=self.tf_logs_path, write_images=True)

            callbacks.append(tensorboard)

        return callbacks

    def get_initial_epoch(self):
        """Get initial_epoch from the last line in the log CSV file."""
        initial_epoch = 0
        if os.path.isfile(self.log_path):
            with open(self.log_path, 'r') as log_file:
                line_ind = 0
                for line_ind, _ in enumerate(log_file):
                    pass
                initial_epoch = line_ind

        return initial_epoch

    def make_data_generator(self, image_folder_dir, validation_mode=False):
        from keras.preprocessing.image import ImageDataGenerator

        # Don't apply randomized data transforms if in validation mode.
        # This will make the validation scores more comparable between epochs.
        if validation_mode:
            generator = ImageDataGenerator(rescale=1. / 255)
        else:
            generator = ImageDataGenerator(
                rescale=1. / 255, horizontal_flip=True, vertical_flip=True)

        generator = generator.flow_from_directory(
            image_folder_dir,
            classes=self.options.class_names,
            target_size=(self.options.input_size, self.options.input_size),
            batch_size=self.options.batch_size,
            class_mode='categorical')

        return generator

    def train(self, do_monitoring):
        loss_function = 'categorical_crossentropy'
        metrics = ['accuracy']
        initial_epoch = self.get_initial_epoch()
        steps_per_epoch = int(
            self.nb_training_samples / self.options.batch_size)
        validation_steps = int(
            self.nb_validation_samples / self.options.batch_size)

        # Useful for testing
        if self.options.short_epoch:
            steps_per_epoch = 1
            validation_steps = 1

        callbacks = self.make_callbacks(do_monitoring)

        self.model.compile(self.optimizer, loss_function, metrics=metrics)

        if do_monitoring:
            tensorboard_process = Popen(
                ['tensorboard', '--logdir={}'.format(self.tf_logs_path)])
            terminate_at_exit(tensorboard_process)

        self.model.fit_generator(
            self.training_gen,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            epochs=self.options.nb_epochs,
            validation_data=self.validation_gen,
            validation_steps=validation_steps,
            callbacks=callbacks)

        if do_monitoring:
            tensorboard_process.terminate()
