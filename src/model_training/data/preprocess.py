"""
Process data from the 2D semantic labeling for aerial imagery dataset
http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html
to put it into a Keras-friendly format.
"""
from os.path import join

import numpy as np

from .settings import (
    POTSDAM, TRAIN, VALIDATION, seed, get_dataset_info)
from .utils import (
    _makedirs, load_tiff, load_image, rgb_to_label_batch, save_image)
from .generators import save_channel_stats

np.random.seed(seed)


def tile_image(im, size, stride):
    # I'm not sure why, but having the image bigger than 5500 make the tiles
    # in the right-most column have a black stripe in them.
    rows, cols = np.clip(im.shape[0:2], 0, 5500)
    tiles = []

    for row in range(0, rows, stride):
        for col in range(0, cols, stride):
            if row + size <= rows and col + size <= cols:
                tiles.append(im[row:row+size, col:col+size, :])
    return tiles


def process_data(file_indices, raw_rgbir_input_path, raw_depth_input_path,
                 raw_output_path, proc_data_path, get_file_names):
    print('Processing data...')
    _makedirs(proc_data_path)

    for index1, index2 in file_indices:
        print('{}_{}'.format(index1, index2))

        rgbir_file_name, depth_file_name, output_file_name = \
            get_file_names(index1, index2)

        rgbir_input_im = load_tiff(join(raw_rgbir_input_path, rgbir_file_name))

        depth_input_im = load_image(
            join(raw_depth_input_path, depth_file_name))
        depth_input_im = np.expand_dims(depth_input_im, axis=2)

        output_im = load_tiff(join(raw_output_path, output_file_name))
        output_im = np.expand_dims(output_im, axis=0)
        output_im = rgb_to_label_batch(output_im)
        output_im = np.squeeze(output_im, axis=0)
        output_im = np.expand_dims(output_im, axis=2)

        concat_im = np.concatenate(
            [rgbir_input_im, depth_input_im, output_im], axis=2)

        proc_file_name = '{}_{}'.format(index1, index2)
        save_image(join(proc_data_path, proc_file_name), concat_im)


def process_potsdam():
    dataset_info = get_dataset_info(POTSDAM)
    proc_data_path = dataset_info.dataset_path
    raw_data_path = dataset_info.raw_dataset_path

    raw_rgbir_input_path = join(raw_data_path, '4_Ortho_RGBIR')
    raw_depth_input_path = join(raw_data_path, '1_DSM_normalisation')
    raw_output_path = join(raw_data_path, '5_Labels_for_participants')

    def get_file_names(index1, index2):
        rgbir_file_name = 'top_potsdam_{}_{}_RGBIR.tif'.format(index1, index2)
        depth_file_name = \
            'dsm_potsdam_{:0>2}_{:0>2}_normalized_lastools.jpg' \
            .format(index1, index2)
        output_file_name = 'top_potsdam_{}_{}_label.tif'.format(index1, index2)

        return rgbir_file_name, depth_file_name, output_file_name

    train_file_indices, validation_file_inds = dataset_info.get_file_inds()

    train_path = join(proc_data_path, TRAIN)
    process_data(
        train_file_indices, raw_rgbir_input_path, raw_depth_input_path,
        raw_output_path, train_path, get_file_names)

    save_channel_stats(proc_data_path)

    validation_path = join(proc_data_path, VALIDATION)
    process_data(
        validation_file_inds, raw_rgbir_input_path, raw_depth_input_path,
        raw_output_path, validation_path, get_file_names)


if __name__ == '__main__':
    process_potsdam()
