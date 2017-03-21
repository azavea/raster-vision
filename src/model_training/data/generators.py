from os.path import join
import argparse

import numpy as np

from .datasets import (POTSDAM, TRAIN, VALIDATION, TEST, PotsdamDataset)
from .settings import datasets_path, results_path
from .utils import (
    load_image, get_image_size, get_channel_stats, plot_sample, _makedirs,
    compute_ndvi, save_numpy_array, save_image)

NUMPY = 'numpy'
IMAGE = 'image'
PROCESSED_POTSDAM = 'processed_potsdam'


class Generator():
    def make_split_generator(self, split, tile_size=None, batch_size=32,
                             shuffle=False, augment=False, normalize=False,
                             eval_mode=False):
        pass


class FileGenerator(Generator):
    """
    A generic data generator that creates batches from files. It can read
    windows of data from disk without loading the entire file into memory.
    """
    def __init__(self):
        print('Computing dataset stats...')
        gen = self.make_split_generator(
            TRAIN, tile_size=(10, 10), batch_size=100, shuffle=True,
            augment=False, normalize=False)
        inputs, _ = next(gen)
        self.normalize_params = get_channel_stats(inputs)
        print('Done.')

    def get_samples(self, gen, nb_samples):
        samples = []
        file_inds = []
        for i, (sample, file_ind) in enumerate(gen):
            samples.append(np.expand_dims(sample, axis=0))
            file_inds.append(file_ind)
            if i+1 == nb_samples:
                break

        if len(samples) > 0:
            return np.concatenate(samples, axis=0), file_inds
        return None, None

    def make_tile_generator(self, file_inds, tile_size, has_outputs):
        for file_ind in file_inds:
            nb_rows, nb_cols = self.get_file_size(file_ind)

            for row_begin in range(0, nb_rows, tile_size[0]):
                for col_begin in range(0, nb_cols, tile_size[1]):
                    row_end = row_begin + tile_size[0]
                    col_end = col_begin + tile_size[1]
                    if row_end <= nb_rows and col_end <= nb_cols:
                        window = ((row_begin, row_end), (col_begin, col_end))
                        tile = self.get_tile(file_ind, window, has_outputs)

                        yield tile, file_ind

    def make_random_tile_generator(self, file_inds, tile_size, has_outputs):
        nb_files = len(file_inds)

        while True:
            rand_ind = np.random.randint(0, nb_files)
            file_ind = file_inds[rand_ind]

            nb_rows, nb_cols = self.get_file_size(file_ind)

            row_begin = np.random.randint(0, nb_rows - tile_size[0] + 1)
            col_begin = np.random.randint(0, nb_cols - tile_size[1] + 1)
            row_end = row_begin + tile_size[0]
            col_end = col_begin + tile_size[1]

            window = ((row_begin, row_end), (col_begin, col_end))
            tile = self.get_tile(file_ind, window, has_outputs)

            yield tile, file_ind

    def make_batch_generator(self, file_inds, tile_size, batch_size, shuffle,
                             has_outputs):
        def make_gen():
            if shuffle:
                return self.make_random_tile_generator(
                    file_inds, tile_size, has_outputs)
            return self.make_tile_generator(file_inds, tile_size, has_outputs)

        gen = make_gen()
        while True:
            batch, batch_file_inds = self.get_samples(gen, batch_size)
            if batch is None:
                raise StopIteration()

            yield batch, batch_file_inds

    def normalize_inputs(self, batch):
        means, stds = self.normalize_params
        batch = batch - means[np.newaxis, np.newaxis, np.newaxis, :]
        batch = batch / stds[np.newaxis, np.newaxis, np.newaxis, :]
        return batch

    def unnormalize_inputs(self, inputs):
        means, stds = self.normalize_params
        nb_dims = len(inputs.shape)
        if nb_dims == 3:
            inputs = np.expand_dims(inputs, 0)

        inputs = inputs * stds[np.newaxis, np.newaxis, np.newaxis, :]
        inputs = inputs + means[np.newaxis, np.newaxis, np.newaxis, :]

        if nb_dims == 3:
            inputs = np.squeeze(inputs, 0)
        return inputs

    def transform_batch(self, batch, augment=False, normalize=False,
                        has_outputs=True):
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

        inputs, outputs, outputs_mask = self.parse_batch(batch, has_outputs)

        if normalize:
            inputs = self.normalize_inputs(inputs)

        return inputs, outputs, outputs_mask

    def make_split_generator(self, split, tile_size=None, batch_size=32,
                             shuffle=False, augment=False, normalize=False,
                             eval_mode=False):
        tile_size = self.dataset.input_shape[0:2] if tile_size is None \
            else tile_size

        file_inds = self.get_file_inds(split)
        has_outputs = split != TEST

        gen = self.make_batch_generator(
            file_inds, tile_size, batch_size, shuffle, has_outputs)

        def transform(x):
            batch, batch_file_inds = x
            inputs, outputs, outputs_mask = self.transform_batch(
                batch, augment=augment, normalize=normalize,
                has_outputs=has_outputs)

            if eval_mode:
                return inputs, outputs, outputs_mask, batch_file_inds
            return inputs, outputs

        gen = map(transform, gen)

        return gen


class PotsdamFileGenerator(FileGenerator):
    """
    A data generator for the Potsdam dataset that creates batches from
    files on disk.
    """
    def __init__(self, include_ir, include_depth, include_ndvi,
                 train_ratio):
        self.dataset = PotsdamDataset(include_ir, include_depth, include_ndvi)
        self.train_ratio = train_ratio

        # The first 17 indices correspond to the training set,
        # and the rest to the validation set used
        # in https://arxiv.org/abs/1606.02585
        self.file_inds = [
            (2, 10), (3, 10), (3, 11), (3, 12), (4, 11), (4, 12), (5, 10),
            (5, 12), (6, 10), (6, 11), (6, 12), (6, 8), (6, 9), (7, 11),
            (7, 12), (7, 7), (7, 9), (2, 11), (2, 12), (4, 10), (5, 11),
            (6, 7), (7, 10), (7, 8)
        ]

        nb_train_inds = int(round(self.train_ratio * len(self.file_inds)))
        self.train_file_inds = self.file_inds[0:nb_train_inds]
        self.validation_file_inds = self.file_inds[nb_train_inds:]

        self.test_file_inds = [
            (2, 13), (2, 14), (3, 13), (3, 14), (4, 13), (4, 14), (4, 15),
            (5, 13), (5, 14), (5, 15), (6, 13), (6, 14), (6, 15), (7, 13)
        ]

        super().__init__()

    def get_file_inds(self, split):
        if split == TRAIN:
            file_inds = self.train_file_inds
        elif split == VALIDATION:
            file_inds = self.validation_file_inds
        elif split == TEST:
            file_inds = self.test_file_inds
        else:
            raise ValueError('{} is not a valid split'.format(split))
        return file_inds


class PotsdamImageFileGenerator(PotsdamFileGenerator):
    """
    A data generator for the Potsdam dataset that creates batches from
    the original TIFF and JPG files.
    """
    def __init__(self, datasets_path, include_ir=False, include_depth=False,
                 include_ndvi=False, train_ratio=0.8):
        self.dataset_path = join(datasets_path, POTSDAM)
        super().__init__(include_ir, include_depth, include_ndvi, train_ratio)

    @staticmethod
    def preprocess(datasets_path):
        # Fix the depth image that is missing a column if it hasn't been
        # fixed already.
        data_path = join(datasets_path, POTSDAM)
        file_path = join(
            data_path,
            '1_DSM_normalisation/dsm_potsdam_03_13_normalized_lastools.jpg')

        im = load_image(file_path)
        if im.shape[1] == 5999:
            im_fix = np.zeros((6000, 6000), dtype=np.uint8)
            im_fix[:, 0:-1] = im[:, :, 0]
            save_image(im_fix, file_path)

    def get_file_size(self, file_ind):
        ind0, ind1 = file_ind

        rgbir_file_path = join(
            self.dataset_path,
            '4_Ortho_RGBIR/top_potsdam_{}_{}_RGBIR.tif'.format(ind0, ind1))
        nb_rows, nb_cols = get_image_size(rgbir_file_path)
        return nb_rows, nb_cols

    def get_tile(self, file_ind, window, has_outputs=True):
        ind0, ind1 = file_ind

        rgbir_file_path = join(
            self.dataset_path,
            '4_Ortho_RGBIR/top_potsdam_{}_{}_RGBIR.tif'.format(ind0, ind1))
        depth_file_path = join(
            self.dataset_path,
            '1_DSM_normalisation/dsm_potsdam_{:0>2}_{:0>2}_normalized_lastools.jpg'.format(ind0, ind1)) # noqa
        outputs_file_path = join(
            self.dataset_path,
            '5_Labels_for_participants/top_potsdam_{}_{}_label.tif'.format(ind0, ind1)) # noqa
        outputs_no_boundary_file_path = join(
            self.dataset_path,
            '5_Labels_for_participants_no_Boundary/top_potsdam_{}_{}_label_noBoundary.tif'.format(ind0, ind1)) # noqa

        rgbir = load_image(rgbir_file_path, window)
        depth = load_image(depth_file_path, window)
        channels = [rgbir, depth]

        if has_outputs:
            outputs = load_image(outputs_file_path, window)
            outputs_no_boundary = load_image(
                outputs_no_boundary_file_path, window)
            channels.extend([outputs, outputs_no_boundary])

        tile = np.concatenate(channels, axis=2)
        return tile

    def parse_batch(self, batch, has_outputs=True):
        rgb = batch[:, :, :, 0:3]
        ir = batch[:, :, :, 3:4]
        depth = batch[:, :, :, 4:5]

        input_channels = [rgb]
        if self.dataset.include_ir:
            input_channels.append(ir)
        if self.dataset.include_depth:
            input_channels.append(depth)
        if self.dataset.include_ndvi:
            red = rgb[:, :, :, 0:1]
            ndvi = compute_ndvi(red, ir)
            input_channels.append(ndvi)

        inputs = np.concatenate(input_channels, axis=3)

        outputs = None
        outputs_mask = None
        if has_outputs:
            outputs = self.dataset.rgb_to_one_hot_batch(batch[:, :, :, 5:8])
            outputs_mask = self.dataset.rgb_to_mask_batch(batch[:, :, :, 8:])
        return inputs, outputs, outputs_mask


class PotsdamNumpyFileGenerator(PotsdamFileGenerator):
    """
    A data generator for the Potsdam dataset that creates batches from
    numpy array files. This is about 20x faster than reading the raw files.
    """
    def __init__(self, datasets_path, include_ir=False, include_depth=False,
                 include_ndvi=False, train_ratio=0.8):
        self.raw_dataset_path = join(datasets_path, POTSDAM)
        self.dataset_path = join(datasets_path, PROCESSED_POTSDAM)
        super().__init__(include_ir, include_depth, include_ndvi, train_ratio)

    @staticmethod
    def preprocess(datasets_path):
        proc_data_path = join(datasets_path, PROCESSED_POTSDAM)
        _makedirs(proc_data_path)

        generator = PotsdamImageFileGenerator(
            datasets_path, include_ir=True, include_depth=True,
            include_ndvi=False)
        dataset = generator.dataset
        full_tile_size = dataset.full_tile_size

        def _preprocess(split):
            gen = generator.make_split_generator(
                split, tile_size=(full_tile_size, full_tile_size),
                batch_size=1, shuffle=False, augment=False, normalize=False,
                eval_mode=True)

            for inputs, outputs, outputs_mask, file_inds in gen:
                file_ind = file_inds[0]

                inputs = np.squeeze(inputs, axis=0)
                channels = [inputs]

                if outputs is not None:
                    outputs = np.squeeze(outputs, axis=0)
                    outputs = dataset.one_hot_to_label_batch(outputs)
                    outputs_mask = np.squeeze(outputs_mask, axis=0)
                    channels.extend([outputs, outputs_mask])
                channels = np.concatenate(channels, axis=2)

                ind0, ind1 = file_ind
                file_name = '{}_{}'.format(ind0, ind1)
                save_numpy_array(
                    join(proc_data_path, file_name), channels)

                # Free memory
                channels = None
                inputs = None
                outputs = None
                outputs_mask = None

        _preprocess(TRAIN)
        _preprocess(VALIDATION)
        _preprocess(TEST)

    def get_file_path(self, file_ind):
        ind0, ind1 = file_ind
        return join(self.dataset_path, '{}_{}.npy'.format(ind0, ind1))

    def get_file_size(self, file_ind):
        file_path = self.get_file_path(file_ind)
        im = np.load(file_path, mmap_mode='r')
        nb_rows, nb_cols = im.shape[0:2]
        return nb_rows, nb_cols

    def get_tile(self, file_ind, window, has_outputs=True):
        file_path = self.get_file_path(file_ind)
        im = np.load(file_path, mmap_mode='r')
        ((row_begin, row_end), (col_begin, col_end)) = window
        tile = im[row_begin:row_end, col_begin:col_end, :]

        return tile

    def parse_batch(self, batch, has_outputs=True):
        rgb = batch[:, :, :, 0:3]
        ir = batch[:, :, :, 3:4]
        depth = batch[:, :, :, 4:5]

        input_channels = [rgb]
        if self.dataset.include_ir:
            input_channels.append(ir)
        if self.dataset.include_depth:
            input_channels.append(depth)
        if self.dataset.include_ndvi:
            red = rgb[:, :, :, 0:1]
            ndvi = compute_ndvi(red, ir)
            input_channels.append(ndvi)

        inputs = np.concatenate(input_channels, axis=3)
        outputs = None
        outputs_mask = None
        if has_outputs:
            outputs = self.dataset.label_to_one_hot_batch(batch[:, :, :, 5:6])
            outputs_mask = batch[:, :, :, 6:7]
        return inputs, outputs, outputs_mask


def get_data_generator(options, datasets_path):
    if options.dataset_name == POTSDAM:
        if options.generator_name == NUMPY:
            return PotsdamNumpyFileGenerator(
                datasets_path, options.include_ir,
                options.include_depth, options.include_ndvi,
                options.train_ratio)
        elif options.generator_name == IMAGE:
            return PotsdamImageFileGenerator(
                datasets_path, options.include_ir,
                options.include_depth, options.include_ndvi,
                options.train_ratio)
        else:
            raise ValueError('{} is not a valid generator'.format(
                options.generator_name))
    else:
        raise ValueError('{} is not a valid dataset'.format(
            options.dataset_name))


def plot_generator(dataset_name, generator_name, split):
    nb_batches = 2
    batch_size = 4

    class Options():
        def __init__(self):
            self.dataset_name = dataset_name
            self.generator_name = generator_name
            self.include_ir = True
            self.include_depth = True
            self.include_ndvi = True
            self.train_ratio = 0.7

    options = Options()
    generator = get_data_generator(options, datasets_path)

    viz_path = join(
        results_path, 'gen_samples', dataset_name, generator_name, split)
    _makedirs(viz_path)

    gen = generator.make_split_generator(
        TRAIN, batch_size=batch_size, shuffle=True, augment=True,
        normalize=True, eval_mode=True)

    for batch_ind in range(nb_batches):
        inputs, outputs, _, _ = next(gen)
        for sample_ind in range(batch_size):
            file_path = join(
                viz_path, '{}_{}.pdf'.format(batch_ind, sample_ind))
            plot_sample(
                file_path, inputs[sample_ind, :, :, :],
                outputs[sample_ind, :, :, :], generator)


def preprocess():
    PotsdamImageFileGenerator.preprocess(datasets_path)
    PotsdamNumpyFileGenerator.preprocess(datasets_path)


def plot_generators():
    plot_generator(POTSDAM, NUMPY, TRAIN)
    plot_generator(POTSDAM, NUMPY, VALIDATION)

    plot_generator(POTSDAM, IMAGE, TRAIN)
    plot_generator(POTSDAM, IMAGE, VALIDATION)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--preprocess',
        action='store_true', help='run preprocessing for all generators')
    parser.add_argument(
        '--plot',
        action='store_true', help='plot all generators')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.preprocess:
        preprocess()

    if args.plot:
        plot_generators()
