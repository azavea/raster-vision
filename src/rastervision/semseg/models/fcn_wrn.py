"""
WideResNet based FCN.
"""
from keras.models import Model
from keras.layers import (
    Input, Activation, Reshape, Conv2D, Lambda, Add)
import tensorflow as tf

from rastervision.common.models.wideresnet import WideResidualNetwork


FCN_WRN = 'fcn_wrn'


def make_fcn_wrn(input_shape, nb_labels, use_pretraining=False, activation='sigmoid'):
    nb_rows, nb_cols, _ = input_shape
    input_tensor = Input(shape=input_shape)
    weights = 'cifar10' if use_pretraining else None

    model = WideResidualNetwork(
        include_top=False, weights=weights, input_tensor=input_tensor)

    x256 = model.get_layer('add_4').output
    x128 = model.get_layer('add_8').output
    x64 = model.get_layer('add_12').output

    c256 = Conv2D(nb_labels, (1, 1), name='conv_labels_256')(x256)
    c128 = Conv2D(nb_labels, (1, 1), name='conv_labels_128')(x128)
    c64 = Conv2D(nb_labels, (1, 1), name='conv_labels_64')(x64)

    def resize_bilinear(images):
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])

    r256 = Lambda(resize_bilinear, name='resize_labels_256')(c256)
    r128 = Lambda(resize_bilinear, name='resize_labels_128')(c128)
    r64 = Lambda(resize_bilinear, name='resize_labels_64')(c64)

    m = Add(name='merge_labels')([r256, r128, r64])

    x = Reshape((nb_rows * nb_cols, nb_labels))(m)
    x = Activation(activation)(x)
    x = Reshape((nb_rows, nb_cols, nb_labels))(x)

    model = Model(inputs=input_tensor, outputs=x)

    return model
