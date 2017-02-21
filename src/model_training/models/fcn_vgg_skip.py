"""
Train a Fully Convolutional Network (FCN) to do semantic labeling. This takes
the convolutional layers of a pretrained VGG model, and adds three skip
connections from feature maps at three different resolutions to the final
classification layer which is similar to FCN-8 in
https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf It also adds
an extra learnable 1x1 convolutional layer for each skip connection which
improves accuracy. Fine-tuning the VGG layers might have had a similar effect.
This got 78% validation accuracy after 30 GPU minutes of training on RGB data.
"""
from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          Convolution2D,
                          Reshape,
                          Lambda,
                          merge)
from keras.applications.vgg16 import VGG16


def make_fcn_vgg_skip(input_shape, nb_labels):
    nb_rows, nb_cols, _ = input_shape
    nb_labels = nb_labels

    input_tensor = Input(shape=input_shape)
    base_model = VGG16(include_top=False, weights='imagenet',
                       input_tensor=input_tensor)
    for layer in base_model.layers:
        layer.trainable = False

    x64 = base_model.get_layer('block3_conv3').output
    x32 = base_model.get_layer('block4_conv3').output
    x16 = base_model.get_layer('block5_conv3').output

    c64 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(x64)
    l64 = Convolution2D(nb_labels, 1, 1, border_mode='same')(c64)

    c32 = Convolution2D(128, 1, 1, border_mode='same', activation='relu')(x32)
    l32 = Convolution2D(nb_labels, 1, 1, border_mode='same')(c32)

    c16 = Convolution2D(256, 1, 1, border_mode='same', activation='relu')(x16)
    l16 = Convolution2D(nb_labels, 1, 1, border_mode='same')(c16)

    def resize_bilinear(images):
        # Workaround for
        # https://github.com/fchollet/keras/issues/4609
        import tensorflow as tf
        nb_rows = 512
        nb_cols = 512
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])

    b64 = Lambda(resize_bilinear)(l64)
    b32 = Lambda(resize_bilinear)(l32)
    b16 = Lambda(resize_bilinear)(l16)

    x = merge([b64, b32, b16], mode='sum')

    x = Reshape([nb_rows * nb_cols, nb_labels])(x)
    x = Activation('softmax')(x)
    x = Reshape([nb_rows, nb_cols, nb_labels])(x)

    model = Model(input=input_tensor, output=x)

    return model
