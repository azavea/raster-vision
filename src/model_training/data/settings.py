from os.path import join, isfile
from os import listdir

RGB_INPUT = 'rgb_input'
DEPTH_INPUT = 'depth_input'
OUTPUT = 'output'
TRAIN = 'train'
VALIDATION = 'validation'
BIG_VALIDATION = 'big_validation'
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

big_tile_size = 2000
big_target_size = (big_tile_size, big_tile_size)

seed = 1


def get_dataset_path(dataset):
    if dataset == VAIHINGEN:
        return processed_vaihingen_path
    elif dataset == POTSDAM:
        return processed_potsdam_path
    else:
        raise ValueError('{} is not a valid dataset'.format(dataset))


def get_nb_validation_samples(data_path, use_big_tiles=False):
    partition_name = BIG_VALIDATION if use_big_tiles else 'validation'
    validation_path = join(data_path, partition_name, RGB_INPUT, BOGUS_CLASS)
    nb_files = 0
    for file_name in listdir(validation_path):
        if isfile(join(validation_path, file_name)):
            nb_files += 1

    return nb_files


def get_input_shape(include_depth, is_big):
    nb_channels = 4 if include_depth else 3
    if is_big:
        return (big_tile_size, big_tile_size, nb_channels)
    else:
        return (tile_size, tile_size, nb_channels)
