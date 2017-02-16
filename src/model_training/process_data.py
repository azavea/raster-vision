"""
Process data from the 2D semantic labeling for aerial imagery dataset
http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html
to put it into a Keras-friendly format.
"""
from os.path import join
from os import listdir, makedirs
import random
import re

import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

RGB_INPUT = 'rgb_input'
DEPTH_INPUT = 'depth_input'
OUTPUT = 'output'
TRAIN = 'train'
VALIDATION = 'validation'
BOGUS_CLASS = 'bogus_class'

train_ratio = 0.75
tile_size = 256
tile_stride = int(tile_size / 2)
target_size = (tile_size, tile_size)
seed = 1

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
raw_data_path = join(datasets_path, 'ISPRS_semantic_labeling_Vaihingen')
raw_rgb_input_path = join(raw_data_path, 'top')
raw_depth_input_path = join(raw_data_path, 'dsm')
raw_output_path = join(raw_data_path, 'gts_for_participants')
proc_data_path = join(datasets_path, 'processed_vaihingen')
results_path = join(data_path, 'results')


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
    rows, cols = im.shape[0:2]
    tiles = []
    for row in range(0, rows, stride):
        for col in range(0, cols, stride):
            if row + size <= rows and col + size <= cols:
                tiles.append(im[row:row+size, col:col+size, :])
    return tiles


def process_data():
    print('Processing data...')
    file_names = [file_name for file_name in listdir(raw_output_path)
                  if file_name.endswith('.tif')]
    random.shuffle(file_names)
    nb_files = len(file_names)

    nb_train_files = int(nb_files * train_ratio)
    train_file_names = file_names[0:nb_train_files]
    validation_file_names = file_names[nb_train_files:]

    file_name_re = re.compile('.*area(\d+).tif')

    def _process_data(file_names, partition_name):
        # Keras expects a directory for each class, but there are none,
        # so put all images in a single bogus class directory.
        proc_rgb_input_path = join(proc_data_path, partition_name,
                                   RGB_INPUT, BOGUS_CLASS)
        proc_depth_input_path = join(proc_data_path, partition_name,
                                     DEPTH_INPUT, BOGUS_CLASS)
        proc_output_path = join(proc_data_path, partition_name,
                                OUTPUT, BOGUS_CLASS)

        _makedirs(proc_rgb_input_path)
        _makedirs(proc_depth_input_path)
        _makedirs(proc_output_path)

        proc_file_index = 0
        for file_name in file_names:
            output_im = load_image(join(raw_output_path, file_name))
            rgb_input_im = load_image(join(raw_rgb_input_path, file_name))
            index = file_name_re.search(file_name).group(1)
            depth_file_name = 'dsm_09cm_matching_area{}.tif'.format(index)
            depth_input_im = load_image(
                join(raw_depth_input_path, depth_file_name))[:, :, np.newaxis]

            rgb_input_tiles = tile_image(rgb_input_im, tile_size, tile_stride)
            depth_input_tiles = tile_image(depth_input_im, tile_size,
                                           tile_stride)
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

    if augment:
        gen = map(rand_rotate_batch, gen)

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
    output_path = join(base_path, OUTPUT)
    output_gen = make_data_generator(output_path, batch_size=batch_size,
                                     shuffle=True, augment=True, one_hot=True)

    return zip(input_gen, output_gen)


def make_input_output_generators(batch_size, include_depth=False):
    train_gen = \
        make_input_output_generator(
            join(proc_data_path, TRAIN), batch_size, include_depth)
    validation_gen = \
        make_input_output_generator(
            join(proc_data_path, VALIDATION), batch_size, include_depth)

    return train_gen, validation_gen


if __name__ == '__main__':
    process_data()
