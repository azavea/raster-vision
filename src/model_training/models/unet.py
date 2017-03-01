"""
A model inspired by U-Net.
https://arxiv.org/pdf/1505.04597
Code was modified from
https://github.com/jocicmarko/ultrasound-nerve-segmentation
"""

from keras.models import Model
from keras.layers import (Input, merge, Convolution2D, MaxPooling2D,
    UpSampling2D, Activation, Reshape, BatchNormalization)


def make_conv_block(nb_filters, input_tensor):
    x = Convolution2D(nb_filters, 3, 3, activation='relu',
                      border_mode='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filters, 3, 3, activation='relu',
                      border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def make_unet(input_shape, nb_labels):
    nb_rows, nb_cols, _ = input_shape

    inputs = Input(input_shape)
    conv1 = make_conv_block(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = make_conv_block(64, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = make_conv_block(128, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = make_conv_block(256, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = make_conv_block(512, pool4)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat')
    conv6 = make_conv_block(256, up6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat')
    conv7 = make_conv_block(128, up7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat')
    conv8 = make_conv_block(64, up8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat')
    conv9 = make_conv_block(32, up9)

    conv10 = Convolution2D(nb_labels, 1, 1)(conv9)

    output = Reshape((nb_rows * nb_cols, nb_labels))(conv10)
    output = Activation('softmax')(output)
    output = Reshape((nb_rows, nb_cols, nb_labels))(output)

    model = Model(input=inputs, output=output)

    return model
