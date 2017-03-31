import numpy as np

from .utils import get_channel_stats

TRAIN = 'train'
VALIDATION = 'validation'
TEST = 'test'

NUMPY = 'numpy'
IMAGE = 'image'


class Generator():
    def make_split_generator(self, split, target_size=None, batch_size=32,
                             shuffle=False, augment=False, normalize=False,
                             eval_mode=False):
        """Make a generator for a split of data.

        # Arguments
            split: a string with the name of a dataset split (eg. train,
                validation, test)
            target_size: tuple of form (nb_rows, nb_cols) with the shape of
                the generated imgs
            batch_size: the size of the minibatches that are generated
            shuffle: True if imgs should be randomly selected from dataset
            augment: True if imgs should be randomly flipped and rotated
            normalize: True if imgs should be shifted and scaled
            eval_mode: True if file_inds and batch_y_mask should be returned

        # Returns
            Returns a Python generator. If eval_mode == True, the generator
            returns a tuple of form
            (batch_x, batch_y, batch_y_mask, batch_file_inds). batch_x is of
            form (batch_size, nb_rows, nb_cols, nb_channels),
            batch_y is one-hot
            coded and is of form (batch_size, nb_rows, nb_cols, nb_labels),
            batch_y_mask is of form (batch_size, nb_rows, nb_cols) and is True
            if that pixel should be used in the final evaluation,
            batch_file_inds is a list of length batch_size and has the
            indices of the files used to generate that batch. If
            eval_mode == False, the batch_y_mask and batch_file_inds are
            omitted. batch_y and batch_y_mask are None when the split has no
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
            TRAIN, target_size=(10, 10), batch_size=100, shuffle=True,
            augment=False, normalize=False, eval_mode=True)
        _, _, all_batch_x, _, _ = next(gen)
        self.normalize_params = get_channel_stats(all_batch_x)
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

    def make_img_generator(self, file_inds, target_size, has_y):
        for file_ind in file_inds:
            nb_rows, nb_cols = self.get_file_size(file_ind)

            if target_size is None:
                window = ((0, nb_rows), (0, nb_cols))
                img = self.get_img(file_ind, window, has_y)
                yield img, file_ind
            else:
                for row_begin in range(0, nb_rows, target_size[0]):
                    for col_begin in range(0, nb_cols, target_size[1]):
                        row_end = row_begin + target_size[0]
                        col_end = col_begin + target_size[1]
                        if row_end <= nb_rows and col_end <= nb_cols:
                            window = ((row_begin, row_end),
                                      (col_begin, col_end))
                            img = self.get_img(file_ind, window, has_y)
                            yield img, file_ind

    def make_random_img_generator(self, file_inds, target_size, has_y):
        nb_files = len(file_inds)

        while True:
            rand_ind = np.random.randint(0, nb_files)
            file_ind = file_inds[rand_ind]

            nb_rows, nb_cols = self.get_file_size(file_ind)

            row_begin = np.random.randint(0, nb_rows - target_size[0] + 1)
            col_begin = np.random.randint(0, nb_cols - target_size[1] + 1)
            row_end = row_begin + target_size[0]
            col_end = col_begin + target_size[1]

            window = ((row_begin, row_end), (col_begin, col_end))
            img = self.get_img(file_ind, window, has_y)

            yield img, file_ind

    def make_batch_generator(self, file_inds, target_size, batch_size, shuffle,
                             has_y):
        def make_gen():
            if shuffle:
                return self.make_random_img_generator(
                    file_inds, target_size, has_y)
            return self.make_img_generator(file_inds, target_size, has_y)

        gen = make_gen()
        while True:
            batch, batch_file_inds = self.get_samples(gen, batch_size)
            if batch is None:
                raise StopIteration()

            yield batch, batch_file_inds

    def normalize(self, batch_x):
        means, stds = self.normalize_params
        batch_x = batch_x - means[np.newaxis, np.newaxis, np.newaxis, :]
        batch_x = batch_x / stds[np.newaxis, np.newaxis, np.newaxis, :]
        return batch_x

    def unnormalize(self, batch_x):
        means, stds = self.normalize_params
        nb_dims = len(batch_x.shape)
        if nb_dims == 3:
            batch_x = np.expand_dims(batch_x, 0)

        batch_x = batch_x * stds[np.newaxis, np.newaxis, np.newaxis, :]
        batch_x = batch_x + means[np.newaxis, np.newaxis, np.newaxis, :]

        if nb_dims == 3:
            batch_x = np.squeeze(batch_x, 0)
        return batch_x

    def make_split_generator(self, split, target_size=None,
                             batch_size=32, shuffle=False, augment=False,
                             normalize=False, eval_mode=False):
        file_inds = self.get_file_inds(split)
        has_y = split != TEST

        gen = self.make_batch_generator(
            file_inds, target_size, batch_size, shuffle, has_y)

        def transform(x):
            batch, batch_file_inds = x
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

            all_batch_x, batch_y, batch_y_mask = self.parse_batch(batch, has_y)
            all_batch_x = self.dataset.augment_channels(all_batch_x)

            if normalize:
                all_batch_x = self.normalize(all_batch_x)

            batch_x = all_batch_x[:, :, :, self.dataset.active_inds]

            if eval_mode:
                return (batch_x, batch_y, all_batch_x, batch_y_mask,
                        batch_file_inds)
            return batch_x, batch_y

        gen = map(transform, gen)

        return gen
