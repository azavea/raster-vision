from os.path import join

RGBIR_INPUT = 'rgbir_input'
DEPTH_INPUT = 'depth_input'
TRAIN = 'train'
VALIDATION = 'validation'
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
    [255, 0, 0],
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
raw_potsdam_path = join(datasets_path, POTSDAM)

tile_size = 256
target_size = (tile_size, tile_size)

big_tile_size = 2000
big_target_size = (big_tile_size, big_tile_size)

seed = 1


class PotsdamInfo():
    def __init__(self):
        self.dataset_path = join(datasets_path, 'processed_potsdam')
        self.raw_dataset_path = join(datasets_path, POTSDAM)

    def get_channel_inds(self, include_ir=False, include_depth=False):
        rgb_input_inds = [0, 1, 2]
        input_inds = [0, 1, 2]
        if include_ir:
            input_inds.append(3)
        if include_depth:
            input_inds.append(4)

        output_inds = [5]

        return rgb_input_inds, input_inds, output_inds

    def get_input_shape(self, include_ir=False, include_depth=False,
                        use_big_tiles=False):
        nb_channels = 3
        if include_ir:
            nb_channels += 1
        if include_depth:
            nb_channels += 1

        if use_big_tiles:
            return (big_tile_size, big_tile_size, nb_channels)
        else:
            return (tile_size, tile_size, nb_channels)

    def get_file_inds():
        # Split used in https://arxiv.org/abs/1606.02585
        training_inds = [
            (2, 10), (3, 10), (3, 11), (3, 12), (4, 11), (4, 12), (5, 10),
            (5, 12), (6, 10), (6, 11), (6, 12), (6, 8), (6, 9), (7, 11),
            (7, 12), (7, 7), (7, 9)
        ]
        validation_inds = [
            (2, 11), (2, 12), (4, 10), (5, 11), (6, 7), (7, 10), (7, 8)
        ]

        return training_inds, validation_inds


def get_dataset_info(dataset):
    if dataset == POTSDAM:
        return PotsdamInfo()
    else:
        raise ValueError('{} is not a valid dataset'.format(dataset))
