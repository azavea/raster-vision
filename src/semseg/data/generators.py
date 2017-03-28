import numpy as np

from .utils import get_channel_stats

TRAIN = 'train'
VALIDATION = 'validation'
TEST = 'test'

NUMPY = 'numpy'
IMAGE = 'image'


class Generator():
    def make_split_generator(self, split, tile_size=None, batch_size=32,
                             shuffle=False, augment=False, normalize=False,
                             eval_mode=False):
        """Make a generator for a split of data.

        # Arguments
            split: a string with the name of a dataset split (eg. train,
                validation, test)
            tile_size: tuple of form (nb_rows, nb_cols) with the shape of
                the generated tiles
            batch_size: the size of the minibatches that are generated
            shuffle: True if tiles should be randomly selected from dataset
            augment: True if tiles should be randomly flipped and rotated
            normalize: True if tiles should be shifted and scaled
            eval_mode: True if file_inds and outputs_masks should be returned

        # Returns
            Returns a Python generator. If eval_mode == True, the generator
            returns a tuple of form
            (inputs, outputs, outputs_mask, batch_file_inds). inputs is of form
            (batch_size, nb_rows, nb_cols, nb_channels), outputs is one-hot
            coded and is of form (batch_size, nb_rows, nb_cols, nb_labels),
            outputs_mask is of form (batch_size, nb_rows, nb_cols) and is True
            if that pixel should be used in the final evaluation,
            batch_file_inds is a list of length batch_size and has the
            indices of the files used to generate that batch. If
            eval_mode == False, the outputs_mask and batch_file_inds are
            omitted. outputs and outputs_mask are None when the split has no
            output images available.
        """
        pass


class FileGenerator(Generator):
    """
    A generic data generator that creates batches from files. It can read
    windows of data from disk without loading the entire file into memory.
    """
    def __init__(self):
        nb_train_inds = int(round(self.train_ratio * len(self.file_inds)))
        self.train_file_inds = self.file_inds[0:nb_train_inds]
        self.validation_file_inds = self.file_inds[nb_train_inds:]

        print('Computing dataset stats...')
        gen = self.make_split_generator(
            TRAIN, tile_size=(10, 10), batch_size=100, shuffle=True,
            augment=False, normalize=False)
        inputs, _ = next(gen)
        self.normalize_params = get_channel_stats(inputs)
        print('Done.')

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

            if tile_size is None:
                window = ((0, nb_rows), (0, nb_cols))
                tile = self.get_tile(file_ind, window, has_outputs)
                yield tile, file_ind
            else:
                for row_begin in range(0, nb_rows, tile_size[0]):
                    for col_begin in range(0, nb_cols, tile_size[1]):
                        row_end = row_begin + tile_size[0]
                        col_end = col_begin + tile_size[1]
                        if row_end <= nb_rows and col_end <= nb_cols:
                            window = ((row_begin, row_end),
                                      (col_begin, col_end))
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

    def make_split_generator(self, split, tile_size=None,
                             batch_size=32, shuffle=False, augment=False,
                             normalize=False, eval_mode=False):
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
