# -*- coding: utf-8 -*-
"""Wide Residual Network models for Keras.
# Reference
- [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers import Input, Conv2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K

TF_WEIGHTS_PATH = 'https://github.com/titu1994/Wide-Residual-Networks/releases/download/v1.2/wrn_28_8_tf_kernels_tf_dim_ordering.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/titu1994/Wide-Residual-Networks/releases/download/v1.2/wrn_28_8_tf_kernels_tf_dim_ordering_no_top.h5'


def WideResidualNetwork(nb_classes, depth=28, width=8, dropout=0.0,
                        include_top=False, weights='cifar10',
                        input_tensor=None, classes=10,
                        input_shape=None, activation='softmax'):
    """Instantiate the Wide Residual Network architecture,
        optionally loading weights pre-trained
        on CIFAR-10. Note that when using TensorFlow,
        for best performance you should set
        `image_dim_ordering="tf"` in your Keras config
        at ~/.keras/keras.json.
        The model and the weights are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            depth: number or layers in the DenseNet
            width: multiplier to the ResNet width (number of filters)
            dropout_rate: dropout rate
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                "cifar10" (pre-training on CIFAR-10)..
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `tf` dim ordering)
                or `(3, 32, 32)` (with `th` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        """

    if weights not in {'cifar10', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-10).')

    if weights == 'cifar10' and include_top and classes != 10:
        raise ValueError('If using `weights` as CIFAR 10 with `include_top`'
                         ' as true, `classes` should be 10')

    if (depth - 4) % 6 != 0:
        raise ValueError('Depth of the network must be such that (depth - 4)'
                         'should be divisible by 6.')

    N = (depth - 4) // 6

    if input_tensor is None:
        input_tensor = Input(shape=input_shape)

    x = __conv1_block(input_tensor)
    nb_conv = 4

    for i in range(N):
        x = __conv2_block(x, i, width, dropout)
        nb_conv += 2

    x = MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = __conv3_block(x, i, width, dropout)
        nb_conv += 2

    x = MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = ___conv4_block(x, i, width, dropout)
        nb_conv += 2

    x = AveragePooling2D((8, 8))(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(nb_classes, activation='softmax')(x)

    # Create model.
    model = Model(input_tensor, x, name='wideresnet')

    # load weights
    if weights == 'cifar10':
        if (depth == 28) and (width == 8) and (dropout == 0.0):
            # Default parameters match. Weights for this model exist:
            if include_top:
                weights_path = get_file('wide_resnet_28_8_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('wide_resnet_28_8_tf_dim_ordering_tf_kernels_no_top.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')

            model.load_weights(weights_path)
    return model


def __conv1_block(input):
    conv_name_base = 'res_a_branch'
    bn_name_base = 'bn_a_branch'
    act_name = 'act_a'

    x = Conv2D(16, (3, 3), padding='same', name=conv_name_base)(input)

    channel_axis = -1

    x = BatchNormalization(axis=channel_axis, name=bn_name_base)(x)
    x = Activation('relu', name=act_name)(x)
    return x


def __conv2_block(input, block, k=1, dropout=0.0):
    conv_name_base = 'res' + str(block) + '_b' + '_branch'
    bn_name_base = 'bn' + str(block) + '_b' + '_branch'
    act_name = 'act' + str(block) + '_b'
    init = input

    channel_axis = -1

    if init._keras_shape[-1] != 16 * k:
        init = Conv2D(16 * k, (1, 1), activation='linear', padding='same', name=conv_name_base + '.1')(init)

    x = Conv2D(16 * k, (3, 3), padding='same', name=conv_name_base + '.2')(input)
    x = BatchNormalization(axis=channel_axis, name=bn_name_base + '.1')(x)
    x = Activation('relu', name=act_name + '.1')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(16 * k, (3, 3), padding='same', name=conv_name_base + '.3')(x)
    x = BatchNormalization(axis=channel_axis, name=bn_name_base + '.2')(x)
    x = Activation('relu', name=act_name + '.2')(x)

    m = add([init, x])
    return m


def __conv3_block(input, block, k=1, dropout=0.0):
    conv_name_base = 'res' + str(block) + '_c' + '_branch'
    bn_name_base = 'bn' + str(block) + '_c' + '_branch'
    act_name = 'act' + str(block) + '_c'
    init = input

    channel_axis = -1

    if init._keras_shape[-1] != 32 * k:
        init = Conv2D(32 * k, (1, 1), activation='linear', padding='same', name=conv_name_base + '.1')(init)

    x = Conv2D(32 * k, (3, 3), padding='same', name=conv_name_base + '.2')(input)
    x = BatchNormalization(axis=channel_axis, name=bn_name_base + '.1')(x)
    x = Activation('relu', name=act_name + '.1')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(32 * k, (3, 3), padding='same', name=conv_name_base + '.3')(x)
    x = BatchNormalization(axis=channel_axis, name=bn_name_base + '.2')(x)
    x = Activation('relu', name=act_name + '.2')(x)

    m = add([init, x])
    return m


def ___conv4_block(input, block, k=1, dropout=0.0):
    conv_name_base = 'res' + str(block) + '_d' + '_branch'
    bn_name_base = 'bn' + str(block) + '_d' + '_branch'
    act_name = 'act' + str(block) + '_d'
    init = input

    channel_axis = -1

    if init._keras_shape[-1] != 64 * k:
        init = Conv2D(64 * k, (1, 1), activation='linear', padding='same', name=conv_name_base + '.1')(init)

    x = Conv2D(64 * k, (3, 3), padding='same', name=conv_name_base + '.2')(input)
    x = BatchNormalization(axis=channel_axis, name=bn_name_base + '.1')(x)
    x = Activation('relu', name=act_name + '.1')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(64 * k, (3, 3), padding='same', name=conv_name_base + '.3')(x)
    x = BatchNormalization(axis=channel_axis, name=bn_name_base + '.2')(x)
    x = Activation('relu', name=act_name + '.2')(x)

    m = add([init, x])
    return m
