from os.path import join
import json

import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from .settings import (
    TRAIN, POTSDAM, get_dataset_info)
from .utils import label_to_one_hot_batch, one_hot_to_rgb_batch, _makedirs


def get_samples(gen, nb_samples):
    samples = []
    for i, sample in enumerate(gen):
        samples.append(np.expand_dims(sample, axis=0))
        if i+1 == nb_samples:
            break

    if len(samples) > 0:
        return np.concatenate(samples, axis=0)
    return None


def get_channel_stats(path, file_names):
    tile_size = (10, 10)
    nb_samples = 2000
    tile_gen = make_random_tile_generator(path, file_names, tile_size)
    samples = get_samples(tile_gen, nb_samples)
    nb_channels = samples.shape[3]
    channel_data = np.reshape(
        np.transpose(samples, [3, 0, 1, 2]), (nb_channels, -1))

    means = np.mean(channel_data, axis=1)
    stds = np.std(channel_data, axis=1)
    return means, stds


def save_channel_stats(path, means, stds):
    channel_stats = map(lambda x: {'mean': x[0], 'std': x[1]},
                        zip(means, stds))
    channel_stats_json = json.dumps({'stats': list(channel_stats)}, indent=4)
    with open(join(path, 'stats.txt'), 'w') as stats_file:
        stats_file.write(channel_stats_json)


def load_channel_stats(path):
    with open(join(path, 'stats.txt'), 'r') as stats_file:
        stats_json = stats_file.read()
        stats = json.loads(stats_json)
        means = np.array(list(map(lambda x: x['mean'], stats['stats'])))
        stds = np.array(list(map(lambda x: x['std'], stats['stats'])))

        return means, stds


def make_tile_generator(path, file_names, tile_size):
    for file_name in file_names:
        file_path = join(path, file_name)
        concat_im = np.load(file_path, mmap_mode='r')
        nb_rows, nb_cols, _ = concat_im.shape

        for row_begin in range(0, nb_rows, tile_size[0]):
            for col_begin in range(0, nb_cols, tile_size[1]):
                row_end = row_begin + tile_size[0]
                col_end = col_begin + tile_size[1]
                if row_end <= nb_rows and col_end <= nb_cols:
                    tile = concat_im[row_begin:row_end, col_begin:col_end, :]
                    # Make writeable in-memory copy
                    tile = np.array(tile)
                    yield tile


def make_random_tile_generator(path, file_names, tile_size):
    nb_files = len(file_names)

    while True:
        file_ind = np.random.randint(0, nb_files)

        file_path = join(path, file_names[file_ind])
        concat_im = np.load(file_path, mmap_mode='r')
        nb_rows, nb_cols, _ = concat_im.shape

        row_begin = np.random.randint(0, nb_rows - tile_size[0] + 1)
        col_begin = np.random.randint(0, nb_cols - tile_size[1] + 1)
        row_end = row_begin + tile_size[0]
        col_end = col_begin + tile_size[1]

        tile = concat_im[row_begin:row_end, col_begin:col_end, :]
        # Make writeable in-memory copy
        tile = np.array(tile)
        yield tile


def make_batch_generator(path, file_names, tile_size, batch_size, shuffle,
                         reset_interval):
    def make_gen():
        if shuffle:
            return make_random_tile_generator(path, file_names, tile_size)
        return make_tile_generator(path, file_names, tile_size)

    gen = make_gen()
    sample_count = 0
    while True:
        samples = get_samples(gen, batch_size)
        sample_count += batch_size
        if reset_interval is not None and sample_count >= reset_interval:
            gen = make_gen()
        if samples is None:
            raise StopIteration()
        yield samples


def transform_batch(batch, dataset_info, augment=False, scale_params=None,
                    eval_mode=False):
    batch = batch.astype(np.float32)

    if augment:
        nb_rotations = np.random.randint(0, 4)

        batch = np.transpose(batch, [1, 2, 3, 0])
        batch = np.rot90(batch, nb_rotations)
        batch = np.transpose(batch, [3, 0, 1, 2])

        if np.random.uniform() > 0.5:
            batch = np.flip(batch, axis=1)
        if np.random.uniform() > 0.5:
            batch = np.flip(batch, axis=2)

    if scale_params is not None:
        means, stds = scale_params
        batch[:, :, :, dataset_info.input_inds] -= \
            means[np.newaxis, np.newaxis, np.newaxis, dataset_info.input_inds]
        batch[:, :, :, dataset_info.input_inds] /= \
            stds[np.newaxis, np.newaxis, np.newaxis, dataset_info.input_inds]

    inputs = batch[:, :, :, dataset_info.input_inds]

    if eval_mode:
        outputs = batch[:, :, :, dataset_info.output_inds]
        outputs_mask = batch[:, :, :, dataset_info.output_mask_inds]
        return inputs, outputs, outputs_mask
    else:
        outputs = batch[:, :, :, dataset_info.output_inds]
        outputs = np.squeeze(outputs, axis=3)
        outputs = label_to_one_hot_batch(outputs)
        return inputs, outputs


def make_split_generator(dataset_info, split, batch_size=32, shuffle=False,
                         reset_interval=None, augment=False, scale=False,
                         eval_mode=False):
    path = dataset_info.dataset_path

    scale_params = load_channel_stats(path) \
        if scale else None

    tile_size = dataset_info.input_shape[0:2]
    file_names = dataset_info.train_file_names if split == TRAIN \
        else dataset_info.validation_file_names
    gen = make_batch_generator(
        path, file_names, tile_size, batch_size, shuffle, reset_interval)

    def transform(batch):
        return transform_batch(
            batch, dataset_info, augment=augment, scale_params=scale_params,
            eval_mode=eval_mode)
    gen = map(transform, gen)

    return gen


def unscale_inputs(inputs, input_inds, scale_params):
    means, stds = scale_params
    nb_dims = len(inputs.shape)
    if nb_dims == 3:
        inputs = np.expand_dims(inputs, 0)

    inputs = inputs * stds[np.newaxis, np.newaxis, np.newaxis, input_inds]
    inputs = inputs + means[np.newaxis, np.newaxis, np.newaxis, input_inds]

    if nb_dims == 3:
        inputs = np.squeeze(inputs, 0)
    return inputs


def plot_sample(file_path, inputs, outputs, dataset_info,
                scale_params):
    inputs = unscale_inputs(inputs, dataset_info.input_inds, scale_params)

    fig = plt.figure()
    nb_input_inds = inputs.shape[2]
    nb_output_inds = outputs.shape[2]

    gs = mpl.gridspec.GridSpec(2, 7)

    def plot_image(plot_row, plot_col, im, is_rgb=False):
        a = fig.add_subplot(gs[plot_row, plot_col])
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)

        if is_rgb:
            a.imshow(im.astype(np.uint8))
        else:
            a.imshow(im, cmap='gray', vmin=0, vmax=255)

    plot_row = 0
    plot_col = 0
    im = inputs[:, :, dataset_info.rgb_input_inds]
    plot_image(plot_row, plot_col, im, is_rgb=True)

    for channel_ind in range(nb_input_inds):
        plot_col += 1
        im = inputs[:, :, channel_ind]
        plot_image(plot_row, plot_col, im)

    plot_row = 1
    plot_col = 0
    rgb_outputs = np.squeeze(
        one_hot_to_rgb_batch(np.expand_dims(outputs, axis=0)))
    plot_image(plot_row, plot_col, rgb_outputs, is_rgb=True)

    for channel_ind in range(nb_output_inds):
        plot_col += 1
        im = outputs[:, :, channel_ind] * 150
        plot_image(plot_row, plot_col, im)

    plt.savefig(file_path, bbox_inches='tight', format='pdf', dpi=600)
    plt.close(fig)


def viz_generator():
    dataset = POTSDAM
    nb_batches = 4
    batch_size = 4

    dataset_info = get_dataset_info(dataset)
    dataset_info.setup(
        include_ir=True, include_depth=True, include_ndvi=True,
        train_ratio=dataset_info.sharah_train_ratio)
    path = dataset_info.dataset_path
    viz_path = join(path, 'gen_samples')
    _makedirs(viz_path)

    scale_params = load_channel_stats(path)

    gen = make_split_generator(
        dataset_info, TRAIN, batch_size=batch_size, shuffle=True, augment=True,
        scale=True)

    for batch_ind in range(nb_batches):
        inputs, outputs = next(gen)
        for sample_ind in range(batch_size):
            file_path = join(
                viz_path, '{}_{}.pdf'.format(batch_ind, sample_ind))
            plot_sample(
                file_path, inputs[sample_ind, :, :, :],
                outputs[sample_ind, :, :, :], dataset_info, scale_params)


if __name__ == '__main__':
    viz_generator()
