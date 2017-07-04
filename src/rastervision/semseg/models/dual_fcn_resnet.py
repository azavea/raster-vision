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
import tensorflow as tf

from rastervision.common.models.resnet50 import ResNet50


DUAL_FCN_RESNET = 'dual_fcn_resnet'


def make_dual_fcn_resnet(input_shape, dual_active_input_inds,
                         nb_labels, use_pretraining, freeze_base):
    nb_rows, nb_cols, nb_channels = input_shape
    input_tensor = Input(shape=input_shape)

    # Split input_tensor into two
    def get_input_tensor(it, model_ind):
        channel_tensors = tf.split(it, nb_channels, 3)
        input_tensors = []
        for ind in dual_active_input_inds[model_ind]:
            input_tensors.append(channel_tensors[ind])
        input_tensor = tf.concat(input_tensors, 3)
        return input_tensor

    input_tensor1 = Lambda(lambda x: get_input_tensor(x, 0))(input_tensor)
    input_tensor2 = Lambda(lambda x: get_input_tensor(x, 1))(input_tensor)

    # Train first model using pretraining and freezing as specified
    weights = 'imagenet' if use_pretraining else None
    base_model1 = ResNet50(
        include_top=False, weights=weights, input_tensor=input_tensor1)
    for layer in base_model1.layers:
        layer.name += '_1'
    if freeze_base:
        for layer in base_model1.layers:
            layer.trainable = False

    # Train second model from scratch
    base_model2 = ResNet50(
        include_top=False, weights=None, input_tensor=input_tensor2)
    for layer in base_model2.layers:
        layer.name += '_2'

    x32_1 = base_model1.get_layer('act3d_1').output
    x16_1 = base_model1.get_layer('act4f_1').output
    x8_1 = base_model1.get_layer('act5c_1').output

    x32_2 = base_model2.get_layer('act3d_2').output
    x16_2 = base_model2.get_layer('act4f_2').output
    x8_2 = base_model2.get_layer('act5c_2').output

    x32 = merge([x32_1, x32_2], mode='concat')
    x16 = merge([x16_1, x16_2], mode='concat')
    x8 = merge([x8_1, x8_2], mode='concat')

    c32 = Convolution2D(nb_labels, 1, 1, name='conv_labels_32')(x32)
    c16 = Convolution2D(nb_labels, 1, 1, name='conv_labels_16')(x16)
    c8 = Convolution2D(nb_labels, 1, 1, name='conv_labels_8')(x8)

    def resize_bilinear(images):
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])

    r32 = Lambda(resize_bilinear, name='resize_labels_32')(c32)
    r16 = Lambda(resize_bilinear, name='resize_labels_16')(c16)
    r8 = Lambda(resize_bilinear, name='resize_labels_8')(c8)

    m = merge([r32, r16, r8], mode='sum', name='merge_labels')

    x = Reshape((nb_rows * nb_cols, nb_labels))(m)
    x = Activation('softmax')(x)
    x = Reshape((nb_rows, nb_cols, nb_labels))(x)

    model = Model(input_tensor, output=x)

    return model
