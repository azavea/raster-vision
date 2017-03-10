from os.path import join

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

data_path = '/opt/data/'
datasets_path = join(data_path, 'datasets')
results_path = join(data_path, 'results')


def file_inds_to_name(file_inds):
    return '{}_{}.npy'.format(file_inds[0], file_inds[1])


class PotsdamInfo():
    def __init__(self):
        self.dataset_path = join(datasets_path, 'processed_potsdam')
        self.raw_dataset_path = join(datasets_path, POTSDAM)
        self.small_tile_size = 256
        self.big_tile_size = 2000
        self.nb_labels = len(label_keys)

        # The first 17 indices correspond to the training set,
        # and the rest to the validation set used
        # in https://arxiv.org/abs/1606.02585
        self.all_file_inds = [
            (2, 10), (3, 10), (3, 11), (3, 12), (4, 11), (4, 12), (5, 10),
            (5, 12), (6, 10), (6, 11), (6, 12), (6, 8), (6, 9), (7, 11),
            (7, 12), (7, 7), (7, 9), (2, 11), (2, 12), (4, 10), (5, 11),
            (6, 7), (7, 10), (7, 8)
        ]
        self.all_file_names = list(map(file_inds_to_name, self.all_file_inds))
        self.sharah_train_ratio = 17 / 24

        self.setup()

    def setup_file_names(self):
        nb_train_inds = int(round(self.train_ratio * len(self.all_file_inds)))
        nb_validation_inds = len(self.all_file_inds) - nb_train_inds
        train_inds = self.all_file_inds[0:nb_train_inds]
        validation_inds = self.all_file_inds[nb_train_inds:]
        self.train_file_names = list(map(file_inds_to_name, train_inds))
        self.validation_file_names = \
            list(map(file_inds_to_name, validation_inds))

    def setup_channels(self):
        self.red_ind = 0
        self.green_ind = 1
        self.blue_ind = 2
        self.ir_ind = 3
        self.depth_ind = 4
        self.ndvi_ind = 5

        self.rgb_input_inds = [self.red_ind, self.green_ind, self.blue_ind]
        self.input_inds = list(self.rgb_input_inds)
        if self.include_ir:
            self.input_inds.append(self.ir_ind)
        if self.include_depth:
            self.input_inds.append(self.depth_ind)
        if self.include_ndvi:
            self.input_inds.append(self.ndvi_ind)

        self.output_inds = [6]
        self.output_mask_inds = [7]

        self.nb_channels = len(self.input_inds)
        self.input_shape = (self.small_tile_size, self.small_tile_size, self.nb_channels)
        if self.use_big_tiles:
            self.input_shape = (self.big_tile_size, self.big_tile_size, self.nb_channels)

    def setup(self, include_ir=False, include_depth=False,
              include_ndvi=False, use_big_tiles=False,
              train_ratio=0.8):
        self.include_ir = include_ir
        self.include_depth = include_depth
        self.include_ndvi = include_ndvi
        self.use_big_tiles = use_big_tiles
        self.train_ratio = train_ratio

        self.setup_file_names()
        self.setup_channels()


def get_dataset_info(dataset):
    if dataset == POTSDAM:
        return PotsdamInfo()
    else:
        raise ValueError('{} is not a valid dataset'.format(dataset))
