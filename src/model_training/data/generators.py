from os.path import join

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from .preprocess import (
    RGB_INPUT, DEPTH_INPUT, OUTPUT, TRAIN, VALIDATION, POTSDAM,
    get_dataset_path, seed, target_size, rgb_to_one_hot_batch,
    one_hot_to_rgb_batch)


def rand_rotate_batch(x):
    np.rot90(x, np.random.randint(1, 5))
    return x


def make_data_generator(path, batch_size=32, shuffle=False, augment=False,
                        scale=False, one_hot=False):
    gen_params = {}
    if augment:
        gen_params['horizontal_flip'] = True
        gen_params['vertical_flip'] = True

    if scale:
        gen_params['featurewise_center'] = True
        gen_params['featurewise_std_normalization'] = True
        samples = get_samples_for_fit(path)

    gen = ImageDataGenerator(**gen_params)
    if scale:
        gen.fit(samples)

    gen = gen.flow_from_directory(
        path, class_mode=None, target_size=target_size,
        batch_size=batch_size, shuffle=shuffle, seed=seed)

#    if augment:
#        gen = map(rand_rotate_batch, gen)

    if one_hot:
        gen = map(rgb_to_one_hot_batch, gen)

    return gen


def get_samples_for_fit(path):
    return next(make_data_generator(path, shuffle=True))


def combine_rgb_depth(rgb_depth):
    rgb, depth = rgb_depth
    depth = depth[:, :, :, 0][:, :, :, np.newaxis]
    return np.concatenate([rgb, depth], axis=3)


def make_input_output_generator(base_path, batch_size, include_depth=False):
    rgb_input_path = join(base_path, RGB_INPUT)
    depth_input_path = join(base_path, DEPTH_INPUT)
    output_path = join(base_path, OUTPUT)

    rgb_input_gen = make_data_generator(
        rgb_input_path, batch_size=batch_size, shuffle=True, augment=True,
        scale=True)

    depth_input_gen = make_data_generator(
        depth_input_path, batch_size=batch_size, shuffle=True, augment=True,
        scale=True)

    input_gen = rgb_input_gen
    if include_depth:
        input_gen = map(combine_rgb_depth, zip(rgb_input_gen, depth_input_gen))

    # Don't scale the outputs (because they are labels) and convert to
    # one-hot encoding.
    output_gen = make_data_generator(output_path, batch_size=batch_size,
                                     shuffle=True, augment=True, one_hot=True)

    return zip(input_gen, output_gen)


def make_input_output_generators(base_path, batch_size, include_depth=False):
    train_gen = \
        make_input_output_generator(
            join(base_path, TRAIN), batch_size, include_depth)
    validation_gen = \
        make_input_output_generator(
            join(base_path, VALIDATION), batch_size, include_depth)

    return train_gen, validation_gen


def plot_batch(inputs, outputs, file_path):
    rgb_outputs = one_hot_to_rgb_batch(outputs)
    inputs = np.clip((inputs + 1) * 128, 0, 255)
    outputs = outputs * 128

    fig = plt.figure()
    nb_input_channels = inputs.shape[3]
    nb_output_channels = outputs.shape[3]
    batch_size = inputs.shape[0]
    nb_subplot_cols = nb_input_channels + nb_output_channels + 2
    gs = mpl.gridspec.GridSpec(batch_size, nb_subplot_cols)
    gs.update(wspace=0.1, hspace=0.1, left=0.1, right=0.4, bottom=0.1, top=0.9)

    def plot_image(subplot_index, im, rgb=False):
        a = fig.add_subplot(gs[subplot_index])
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        if rgb:
            a.imshow(im.astype(np.uint8))
        else:
            a.imshow(im, cmap='gray', vmin=0, vmax=255)

    subplot_index = 0
    for batch_ind in range(batch_size):
        # Plot input channels
        for channel_ind in range(nb_input_channels):
            im = inputs[batch_ind, :, :, channel_ind]
            plot_image(subplot_index, im)
            subplot_index += 1

        # Plot RGB input
        im = inputs[batch_ind, :, :, 0:3]
        plot_image(subplot_index, im, rgb=True)
        subplot_index += 1

        # Plot output channels
        for channel_ind in range(nb_output_channels):
            im = outputs[batch_ind, :, :, channel_ind]
            plot_image(subplot_index, im)
            subplot_index += 1

        # Plot output channels jointly
        im = rgb_outputs[batch_ind, :, :, :]
        plot_image(subplot_index, im, rgb=True)
        subplot_index += 1

    plt.savefig(file_path, bbox_inches='tight', format='pdf', dpi=600)


def plot_generators():
    batch_size = 12
    nb_batches = 1
    include_depth = True
    proc_data_path = get_dataset_path(POTSDAM)

    train_gen, validation_gen = \
        make_input_output_generators(proc_data_path, batch_size, include_depth)

    for batch_ind in range(nb_batches):
        inputs, outputs = next(train_gen)
        file_path = join(
            proc_data_path, TRAIN, 'batch_{}.pdf'.format(batch_ind))
        plot_batch(inputs, outputs, file_path)

        inputs, outputs = next(validation_gen)
        file_path = join(
            proc_data_path, VALIDATION, 'batch_{}.pdf'.format(batch_ind))
        plot_batch(inputs, outputs, file_path)


if __name__ == '__main__':
    plot_generators()
