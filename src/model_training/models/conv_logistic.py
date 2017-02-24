"""
An extremely simple model for semantic labeling which is expected to have
poor results. It does logistic regression across sliding windows in the
image. This is just to test that our data is processed correctly and that we
know how to use Keras.
"""
from keras.models import Sequential
from keras.layers import Activation, Convolution2D, Reshape


def make_conv_logistic(input_shape, nb_labels, kernel_size):
    nb_rows, nb_cols, _ = input_shape
    nb_labels = nb_labels

    model = Sequential()
    model.add(Convolution2D(
        nb_labels, kernel_size[0], kernel_size[1], border_mode='same',
        input_shape=input_shape, name='conv_labels'))
    model.add(Reshape([nb_rows * nb_cols, nb_labels]))
    model.add(Activation('softmax'))
    model.add(Reshape([nb_rows, nb_cols, nb_labels]))

    return model
