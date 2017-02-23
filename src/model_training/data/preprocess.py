"""
Process data from the 2D semantic labeling for aerial imagery dataset
http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html
to put it into a Keras-friendly format.
"""
from os.path import join, isfile
from os import listdir, makedirs
import random
import re

import numpy as np
from PIL import Image

RGB_INPUT = 'rgb_input'
DEPTH_INPUT = 'depth_input'
OUTPUT = 'output'
TRAIN = 'train'
VALIDATION = 'validation'
BOGUS_CLASS = 'bogus_class'

VAIHINGEN = 'vaihingen'
POTSDAM = 'potsdam'

# Impervious surfaces (RGB: 255, 255, 255)
# Building (RGB: 0, 0, 255)
# Low vegetation (RGB: 0, 255, 255)
# Tree (RGB: 0, 255, 0)
# Car (RGB: 255, 255, 0)
# Clutter/background (RGB: 255, 0, 0)
label_keys = [
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0]
]
label_names = [
    'Impervious',
    'Building',
    'Low vegetation',
    'Tree',
    'Car',
    'Clutter'
]
nb_labels = len(label_keys)

data_path = '/opt/data/'
datasets_path = join(data_path, 'datasets')
results_path = join(data_path, 'results')
processed_potsdam_path = join(datasets_path, 'processed_potsdam')
processed_vaihingen_path = join(datasets_path, 'processed_vaihingen')

tile_size = 256
target_size = (tile_size, tile_size)

seed = 1
random.seed(seed)


def get_dataset_path(dataset):
    if dataset == VAIHINGEN:
        return processed_vaihingen_path
    elif dataset == POTSDAM:
        return processed_potsdam_path


def get_nb_validation_samples(data_path):
    validation_path = join(data_path, 'validation', RGB_INPUT, BOGUS_CLASS)
    nb_files = 0
    for file_name in listdir(validation_path):
        if isfile(join(validation_path, file_name)):
            nb_files += 1

    return nb_files


def _makedirs(path):
    try:
        makedirs(path)
    except:
        pass


def load_image(file_path):
    im = Image.open(file_path)
    return np.array(im)


def save_image(file_path, im):
    Image.fromarray(np.squeeze(im).astype(np.uint8)).save(file_path)


def rgb_to_label_batch(rgb_batch):
    label_batch = np.zeros(rgb_batch.shape[:-1])
    for label, key in enumerate(label_keys):
        mask = (rgb_batch[:, :, :, 0] == key[0]) & \
               (rgb_batch[:, :, :, 1] == key[1]) & \
               (rgb_batch[:, :, :, 2] == key[2])
        label_batch[mask] = label

    return label_batch


def label_to_one_hot_batch(label_batch):
    one_hot_batch = np.zeros(np.concatenate([label_batch.shape, [nb_labels]]))
    for label in range(nb_labels):
        one_hot_batch[:, :, :, label][label_batch == label] = 1.
    return one_hot_batch


def rgb_to_one_hot_batch(rgb_batch):
    return label_to_one_hot_batch(rgb_to_label_batch(rgb_batch))


def label_to_rgb_batch(label_batch):
    rgb_batch = np.zeros(np.concatenate([label_batch.shape, [3]]))
    for label, key in enumerate(label_keys):
        mask = label_batch == label
        rgb_batch[mask, :] = key

    return rgb_batch


def one_hot_to_label_batch(one_hot_batch):
    return np.argmax(one_hot_batch, axis=3)


def one_hot_to_rgb_batch(one_hot_batch):
    return label_to_rgb_batch(one_hot_to_label_batch(one_hot_batch))


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


def process_data(raw_data_path, raw_rgb_input_path, raw_depth_input_path,
                 raw_output_path, proc_data_path, train_ratio, tile_stride,
                 get_file_names):
    print('Processing data...')
    output_file_names = [file_name for file_name in listdir(raw_output_path)
                         if file_name.endswith('.tif')]
    random.shuffle(output_file_names)
    nb_files = len(output_file_names)

    nb_train_files = int(nb_files * train_ratio)
    train_file_names = output_file_names[0:nb_train_files]
    validation_file_names = output_file_names[nb_train_files:]

    def _process_data(output_file_names, partition_name):
        # Keras expects a directory for each class, but there are none,
        # so put all images in a single bogus class directory.
        proc_rgb_input_path = join(
            proc_data_path, partition_name, RGB_INPUT, BOGUS_CLASS)
        proc_depth_input_path = join(
            proc_data_path, partition_name, DEPTH_INPUT, BOGUS_CLASS)
        proc_output_path = join(
            proc_data_path, partition_name, OUTPUT, BOGUS_CLASS)

        _makedirs(proc_rgb_input_path)
        _makedirs(proc_depth_input_path)
        _makedirs(proc_output_path)

        proc_file_index = 0
        for output_file_name in output_file_names:
            rgb_file_name, depth_file_name = get_file_names(output_file_name)

            output_im = load_image(join(raw_output_path, output_file_name))
            rgb_input_im = load_image(join(raw_rgb_input_path, rgb_file_name))
            depth_input_im = load_image(
                join(raw_depth_input_path, depth_file_name))[:, :, np.newaxis]

            rgb_input_tiles = tile_image(rgb_input_im, tile_size, tile_stride)
            depth_input_tiles = tile_image(
                depth_input_im, tile_size, tile_stride)
            output_tiles = tile_image(output_im, tile_size, tile_stride)

            for rgb_input_tile, depth_input_tile, output_tile in \
                    zip(rgb_input_tiles, depth_input_tiles, output_tiles):
                proc_file_name = '{}.png'.format(proc_file_index)
                save_image(join(proc_rgb_input_path, proc_file_name),
                           rgb_input_tile)
                save_image(join(proc_depth_input_path, proc_file_name),
                           depth_input_tile)
                save_image(join(proc_output_path, proc_file_name),
                           output_tile)
                proc_file_index += 1

    _process_data(train_file_names, TRAIN)
    _process_data(validation_file_names, VALIDATION)


def process_vaihingen():
    raw_data_path = join(datasets_path, 'vaihingen')
    raw_rgb_input_path = join(raw_data_path, 'top')
    raw_depth_input_path = join(raw_data_path, 'dsm')
    raw_output_path = join(raw_data_path, 'gts_for_participants')
    proc_data_path = processed_vaihingen_path

    train_ratio = 0.8
    tile_stride = int(tile_size / 2)

    output_file_name_re = re.compile('.*area(\d+).tif')

    def get_file_names(output_file_name):
        index = output_file_name_re.search(output_file_name).group(1)
        rgb_file_name = output_file_name
        depth_file_name = 'dsm_09cm_matching_area{}.tif'.format(index)

        return rgb_file_name, depth_file_name

    process_data(
        raw_data_path, raw_rgb_input_path, raw_depth_input_path,
        raw_output_path, proc_data_path, train_ratio, tile_stride,
        get_file_names)


def process_potsdam():
    raw_data_path = join(datasets_path, 'potsdam')
    raw_rgb_input_path = join(raw_data_path, '3_Ortho_IRRG')
    raw_depth_input_path = join(raw_data_path, '1_DSM_normalisation')
    raw_output_path = join(raw_data_path, '5_Labels_for_participants')
    proc_data_path = processed_potsdam_path

    train_ratio = 0.8
    tile_stride = 256

    output_file_name_re = re.compile('^top_potsdam_(\d+)_(\d+)_label.tif')

    def get_file_names(output_file_name):
        search = output_file_name_re.search(output_file_name)
        index1 = search.group(1)
        index2 = search.group(2)
        rgb_file_name = 'top_potsdam_{}_{}_IRRG.tif'.format(index1, index2)
        print(rgb_file_name)

        depth_file_name = \
            'dsm_potsdam_{:0>2}_{:0>2}_normalized_lastools.jpg' \
            .format(index1, index2)

        return rgb_file_name, depth_file_name

    process_data(
        raw_data_path, raw_rgb_input_path, raw_depth_input_path,
        raw_output_path, proc_data_path, train_ratio, tile_stride,
        get_file_names)


if __name__ == '__main__':
    # process_vaihingen()
    process_potsdam()
