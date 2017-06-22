# flake8: noqa
# -*- coding: utf-8 -*-
"""WideResNet model for Keras.
# Reference:
- [Wide Residual Networks](https://arxiv.org/abs/1605.07146v1)
Adapted from https://github.com/asmith26/wide_resnets_keras
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import warnings
import sys

from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/asmith26/wide_resnets_keras/raw/master/models/WRN-40-4.h5'

# Prevent reaching to maximum recursion depth in `theano.tensor.grad`
sys.setrecursionlimit(2 ** 20)

# network config
depth = 40
k = 4
dropout_probability = 0
weight_decay = 0.0005
use_bias = False        # following functions 'FCinit(model)' and 'DisableBias(model)' in utils.lua
weight_init="he_normal" # follows the 'MSRinit(model)' function in utils.lua

def _wide_unit(n_input_plane, n_output_plane, stage, block, stride):
    def f(net):
        # format of conv_params:
        #               [ [nb_col="kernel width", nb_row="kernel height",
        #               subsample="(stride_vertical,stride_horizontal)",
        #               border_mode="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [ [3,3,stride,"same"],
                        [3,3,(1,1),"same"] ]

        n_bottleneck_plane = n_output_plane
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + stage + str(block) + '_branch'
        bn_name_base = 'bn' + stage + str(block) + '_branch'
        act_name = 'act' + stage + str(block)

        # Residual block
        for i, v in enumerate(conv_params):
            if i == 0:
                if n_input_plane != n_output_plane:
                    net = BatchNormalization(axis=bn_axis, name=bn_name_base + '1a')(net)
                    net = Activation("relu", name=act_name)(net)
                    convs = net
                else:
                    convs = BatchNormalization(axis=bn_axis, name=bn_name_base + '1a')(net)
                    convs = Activation("relu")(convs)
                convs = Convolution2D(n_bottleneck_plane, nb_col=v[0], nb_row=v[1],
                                     subsample=v[2],
                                     border_mode=v[3],
                                     init=weight_init,
                                     W_regularizer=l2(weight_decay),
                                     bias=use_bias,
                                     name=conv_name_base + '1b')(convs)
            else:
                convs = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(convs)
                convs = Activation("relu")(convs)
                if dropout_probability > 0:
                   convs = Dropout(dropout_probability)(convs)
                convs = Convolution2D(n_bottleneck_plane, nb_col=v[0], nb_row=v[1],
                                     subsample=v[2],
                                     border_mode=v[3],
                                     init=weight_init,
                                     W_regularizer=l2(weight_decay),
                                     bias=use_bias,
                                     name=conv_name_base + '2b')(convs)

        # Shortcut Conntection: identity function or 1x1 convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut = Convolution2D(n_output_plane, nb_col=1, nb_row=1,
                                     subsample=stride,
                                     border_mode="same",
                                     init=weight_init,
                                     W_regularizer=l2(weight_decay),
                                     bias=use_bias,
                                     name=conv_name_base + '1')(net)
        else:
            shortcut = net

        return merge([convs, shortcut], mode="sum")
    return f

# "stacking residual units on the same stage"
def _layer(wide_unit, n_input_plane, n_output_plane, count, stride, stage):
    def f(net):
        block = 1
        net = wide_unit(n_input_plane, n_output_plane, stage, block, stride)(net)
        for i in range(2,int(count+1)):
            block += 1
            net = wide_unit(n_output_plane, n_output_plane, stage, block, stride=(1,1))(net)
        return net
    return f

def WideResNet(include_top=False, weights="CIFAR", input_tensor=None,
               classes=1000, activation='softmax'):
    assert((depth - 4) % 6 == 0)
    n = (depth - 4) / 6

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    n_stages=[16, 16*k, 32*k, 64*k]

    # First conv layer, spatial size 32x32
    conv1 = Convolution2D(nb_filter=n_stages[0], nb_row=3, nb_col=3,
                          subsample=(1, 1),
                          border_mode="same",
                          init=weight_init,
                          W_regularizer=l2(weight_decay),
                          bias=use_bias)(input_tensor)

    # Add wide residual blocks
    block_fn = _wide_unit
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1,1), stage='a')(conv1)# "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2,2), stage='b')(conv2)# "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2,2), stage='c')(conv3)# "Stage 3 (spatial size: 8x8)"

    batch_norm = BatchNormalization(axis=bn_axis)(conv4)
    x = Activation("relu", name="act_final")(batch_norm)

    # Classifier block
    if include_top:
        pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), border_mode="same")(x)
        flatten = Flatten()(pool)
        x = Dense(output_dim=classes, init=weight_init, bias=use_bias,
                            W_regularizer=l2(weight_decay), activation=activation)(flatten)

    # Create model.
    model = Model(input_tensor, x, name="wideresnet")

    # NOTE: this file is for use with theano backend. You must change network
    # architecture to use these weights for pre-training
    if weights == "CIFAR":
        weights_path = get_file('WRN-40-4.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')

        model.load_weights(weights_path, by_name=True)

    return model
