"""
Train a Fully Convolutional Network (FCN) from scratch to do semantic labeling.
It uses a VGG-like architecture and was difficult to train. This was an attempt
at replicating
http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w19/papers/Kampffmeyer_Semantic_Segmentation_of_CVPR_2016_paper.pdf # NOQA
"""
from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          Convolution2D,
                          Reshape,
                          Lambda,
                          BatchNormalization,
                          MaxPooling2D)


def make_fcn(input_shape, nb_labels):
    input_shape = tuple(input_shape)
    nb_rows, nb_cols, _ = input_shape
    nb_labels = nb_labels

    input_tensor = Input(shape=input_shape)

    # Block 1 512x512 -> 64x64
    x = Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2),
                      input_shape=input_shape)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 2 64x64 -> 32x32
    x = Convolution2D(128, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(128, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 3 32x32 -> 16x16
    x = Convolution2D(512, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(512, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 3 16x16 -> 16x16
    x = Convolution2D(512, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(512, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    def resize_bilinear(images):
        # Workaround for
        # https://github.com/fchollet/keras/issues/4609
        import tensorflow as tf
        nb_rows = 512
        nb_cols = 512
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])

    x = Convolution2D(nb_labels, 1, 1)(x)

    x = Lambda(resize_bilinear)(x)

    x = Reshape((nb_rows * nb_cols, nb_labels))(x)
    x = Activation('softmax')(x)
    x = Reshape((nb_rows, nb_cols, nb_labels))(x)

    model = Model(input=input_tensor, output=x)

    return model
