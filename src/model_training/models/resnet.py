# -*- coding: utf-8 -*-
# flake8: noqa
'''ResNet50 model for Keras.
Modified to enable dropout and a smaller sized model to control overfitting.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1256.03385)
Adapted from code contributed by BigMoyan.
Adapted from code from
https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
'''
from __future__ import print_function

import numpy as np
import warnings

from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization, Dropout
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   drop_prob=0.0, last_in_stage=False):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Dropout(drop_prob)(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Dropout(drop_prob)(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)

    if last_in_stage:
        x = Dropout(drop_prob, name='output' + str(stage))(x)
    else:
        x = Dropout(drop_prob)(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),
               drop_prob=0.0):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Dropout(drop_prob)(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Dropout(drop_prob)(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    x = Dropout(drop_prob)(x)

    return x


def ResNet(input_tensor=None, drop_prob=0.0, is_big_model=True):
    img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    filters = [64, 64, 256] if is_big_model else [32, 32, 128]
    x = conv_block(x, 3, filters, stage=2, block='a', strides=(1, 1),
        drop_prob=drop_prob)
    x = identity_block(x, 3, filters, stage=2, block='b',
        drop_prob=drop_prob, last_in_stage=(not is_big_model))
    if is_big_model:
        x = identity_block(x, 3, filters, stage=2, block='c',
            drop_prob=drop_prob, last_in_stage=True)

    filters = [128, 128, 512] if is_big_model else [64, 64, 256]
    x = conv_block(x, 3, filters, stage=3, block='a',
        drop_prob=drop_prob)
    x = identity_block(x, 3, filters, stage=3, block='b',
        drop_prob=drop_prob, last_in_stage=(not is_big_model))
    if is_big_model:
        x = identity_block(x, 3, filters, stage=3, block='c',
            drop_prob=drop_prob)
        x = identity_block(x, 3, filters, stage=3, block='d',
            drop_prob=drop_prob, last_in_stage=True)

    filters = [256, 256, 1024] if is_big_model else [128, 128, 512]
    x = conv_block(x, 3, filters, stage=4, block='a',
        drop_prob=drop_prob)
    x = identity_block(x, 3, filters, stage=4, block='b',
        drop_prob=drop_prob, last_in_stage=(not is_big_model))
    if is_big_model:
        x = identity_block(x, 3, filters, stage=4, block='c',
            drop_prob=drop_prob)
        x = identity_block(x, 3, filters, stage=4, block='d',
            drop_prob=drop_prob)
        x = identity_block(x, 3, filters, stage=4, block='e',
            drop_prob=drop_prob)
        x = identity_block(x, 3, filters, stage=4, block='f',
            drop_prob=drop_prob, last_in_stage=True)

    filters = [512, 512, 2048] if is_big_model else [256, 256, 1024]
    x = conv_block(x, 3, filters, stage=5, block='a',
        drop_prob=drop_prob)
    x = identity_block(x, 3, filters, stage=5, block='b',
        drop_prob=drop_prob, last_in_stage=(not is_big_model))
    if is_big_model:
        x = identity_block(x, 3, filters, stage=5, block='c',
                           drop_prob=drop_prob, last_in_stage=True)

    model = Model(img_input, x)

    return model
