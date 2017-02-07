"""
Train an extremely simple model for semantic labeling which is expected to have
poor results. It's essentially doing logistic regression on each window in the
image. This is just to test that our data is processed correctly and that we
know how to use Keras.
"""
import numpy as np
from os.path import join
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Activation, Convolution2D, Reshape
from keras import backend as K

from process_data import (make_input_output_generators,
                          nb_labels,
                          model_path)

batch_size = 32
samples_per_epoch = 256
nb_epoch = 10
nb_val_samples = 256

train_generator, validation_generator = make_input_output_generators(batch_size)
sample_input, sample_output = next(train_generator)
nb_rows, nb_cols, nb_channels = input_shape = sample_input.shape[1:]

model = Sequential()
nb_filters = nb_labels
kernel_size = (10, 10)
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same', input_shape=input_shape))
model.add(Reshape([nb_rows * nb_cols, nb_filters]))
model.add(Activation('softmax'))
model.add(Reshape([nb_rows, nb_cols, nb_filters]))

print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples)

model.save(join(model_path, 'model.h5'))
