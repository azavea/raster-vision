import os
from pathlib import Path

import keras
from keras.preprocessing.image import ImageDataGenerator

from keras_classification.utils import make_dir


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
        self.log_path = os.path.join(options.output_dir, 'log.csv')
        make_dir(self.log_path, use_dirname=True)

        self.training_gen = self.make_data_generator(
            options.training_data_dir)
        self.validation_gen = self.make_data_generator(
            options.validation_data_dir)

        self.nb_training_samples = get_nb_images(
            options.training_data_dir)
        self.nb_validation_samples = get_nb_images(
            options.validation_data_dir)

    def make_callbacks(self):
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self.model_path, save_best_only=True)

        csv_logger = keras.callbacks.CSVLogger(self.log_path, append=True)

        callbacks = [model_checkpoint, csv_logger]

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

    def make_data_generator(self, image_folder_dir):
        generator = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True)

        generator = generator.flow_from_directory(
            image_folder_dir,
            classes=self.options.class_names,
            target_size=(self.options.input_size, self.options.input_size),
            batch_size=self.options.batch_size,
            class_mode='categorical')

        return generator

    def train(self):
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

        callbacks = self.make_callbacks()

        self.model.compile(self.optimizer, loss_function, metrics=metrics)

        self.model.fit_generator(
            self.training_gen,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            epochs=self.options.nb_epochs,
            validation_data=self.validation_gen,
            validation_steps=validation_steps,
            callbacks=callbacks)
