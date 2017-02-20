"""
Train a Fully Convolutional Network (FCN) to do semantic labeling. This takes
the convolutional layers of a pretrained VGG model, adds a 1x1 convolutional
layer and blows up the 16x16 feature map to 512x512 using bilinear
interpolation. This doesn't have an skip connections so is similar to FCN-32 in
https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
"""
from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          Convolution2D,
                          Reshape,
                          Lambda)
from keras.applications.vgg16 import VGG16


def make_fcn_vgg(input_shape, nb_labels):
    nb_rows, nb_cols, _ = input_shape
    nb_labels = nb_labels

    input_tensor = Input(shape=input_shape)
    base_model = VGG16(include_top=False, weights='imagenet',
                       input_tensor=input_tensor)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.layers[-2].output
    # x = Convolution2D(512, 1, 1, border_mode='same', activation='relu')(x)
    x = Convolution2D(nb_labels, 1, 1, border_mode='same')(x)

    def resize_bilinear(images):
        # Workaround for
        # https://github.com/fchollet/keras/issues/4609
        import tensorflow as tf
        nb_rows = 512
        nb_cols = 512
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])
    x = Lambda(resize_bilinear)(x)

    x = Reshape([nb_rows * nb_cols, nb_labels])(x)
    x = Activation('softmax')(x)
    x = Reshape([nb_rows, nb_cols, nb_labels])(x)

    model = Model(input=input_tensor, output=x)

    return model
