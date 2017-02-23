"""
ResNet based FCN.
"""
from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          Convolution2D,
                          Reshape,
                          Lambda,
                          merge)

from .resnet import ResNet


def make_fcn_resnet(input_shape, nb_labels, drop_prob):
    input_shape = tuple(input_shape)
    nb_rows, nb_cols, _ = input_shape
    nb_labels = nb_labels

    input_tensor = Input(shape=input_shape)
    print('Dropout prob: {}'.format(drop_prob))
    model = ResNet(input_tensor=input_tensor, drop_prob=drop_prob)

    x = model.output

    def resize_bilinear(images):
        # Workaround for
        # https://github.com/fchollet/keras/issues/4609
        import tensorflow as tf
        nb_rows = 256
        nb_cols = 256
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])

    x64 = model.get_layer('activation_22').output
    x32 = model.get_layer('activation_37').output
    x16 = model.get_layer('activation_49').output

    c64 = Convolution2D(nb_labels, 1, 1)(x64)
    c32 = Convolution2D(nb_labels, 1, 1)(x32)
    c16 = Convolution2D(nb_labels, 1, 1)(x16)

    b64 = Lambda(resize_bilinear)(c64)
    b32 = Lambda(resize_bilinear)(c32)
    b16 = Lambda(resize_bilinear)(c16)

    x = merge([b64, b32, b16], mode='sum')

    x = Reshape((nb_rows * nb_cols, nb_labels))(x)
    x = Activation('softmax')(x)
    x = Reshape((nb_rows, nb_cols, nb_labels))(x)

    model = Model(input=input_tensor, output=x)

    return model
