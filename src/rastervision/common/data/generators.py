from os.path import exists, join
import json

import numpy as np
from skimage import transform

from rastervision.common.settings import TRAIN, VALIDATION, TEST
from rastervision.common.utils import get_channel_stats, save_json

ROTATE90 = 'rotate90'
HFLIP = 'hflip'
VFLIP = 'vflip'
ROTATE = 'rotate'
TRANSLATE = 'translate'
all_augment_methods = [ROTATE90, HFLIP, VFLIP, ROTATE, TRANSLATE]
# safe methods don't destroy any information in the original image
safe_augment_methods = [HFLIP, VFLIP, ROTATE90]


class Batch():
    def __init__(self):
        self.file_inds = None
        self.all_x = None
        self.x = None
        self.y = None


class Generator():
    def make_split_generator(self, split, target_size=None, batch_size=32,
                             shuffle=False, augment_methods=None,
                             normalize=False, only_xy=False):
        """Make a generator for a split of data.

        # Arguments
            split: a string with the name of a dataset split (eg. train,
                validation, test)
            target_size: tuple of form (nb_rows, nb_cols) with the shape of
                the generated imgs
            batch_size: the size of the minibatches that are generated
            shuffle: True if imgs should be randomly selected from dataset
            augment_methods: list of augmentation types
            normalize: True if imgs should be shifted and scaled
            only_xy: True if only (x,y) should be returned

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
    def __init__(self, options):
        self.active_input_inds = options.active_input_inds
        self.train_ratio = options.train_ratio
        self.cross_validation = options.cross_validation

        if self.train_ratio is not None:
            nb_train_inds = \
                int(round(self.train_ratio * len(self.dev_file_inds)))
            self.train_file_inds = self.dev_file_inds[0:nb_train_inds]
            self.validation_file_inds = self.dev_file_inds[nb_train_inds:]

        if self.cross_validation is not None:
            self.process_cross_validation()

        self.train_probs = self.compute_split_probs(TRAIN)
        self.validation_probs = self.compute_split_probs(VALIDATION)

        # If a dataset's normalized parameters have already been
        # calculated, load its json file. Otherwise, calculate parameters
        # with a small batch.
        channel_stats_path = join(self.dataset_path,
                                  self.name + '_channel_stats.json')
        if exists(channel_stats_path):
            with open(channel_stats_path) as channel_stats_file:
                channel_stats = json.load(channel_stats_file)
            self.channel_stats = (np.array(channel_stats['means']),
                                  np.array(channel_stats['stds']))
        else:
            self.channel_stats = self.compute_channel_stats(100, False)

    def compute_split_probs(self):
        return None

    def calibrate_image(self, normalized_image):
        calibrated_image = normalized_image.copy()
        for i in range(self.dataset.nb_channels):
            calibrated_image[:, :, i] = \
                (normalized_image[:, :, i] * self.dataset.display_stds[i] +
                 self.dataset.display_means[i])
        calibrated_image = np.clip(calibrated_image, 0, 1)
        return calibrated_image

    def has_y(self, file_ind):
        return file_ind in self.dev_file_inds

    def process_cross_validation(self):
        fold_sizes = self.cross_validation['fold_sizes']
        fold_ind = self.cross_validation['fold_ind']

        fold_ends = list(np.cumsum(fold_sizes))
        fold_start = 0
        for curr_fold_ind, fold_end in enumerate(fold_ends):
            fold_file_inds = self.dev_file_inds[fold_start:fold_end]
            if fold_ind == curr_fold_ind:
                break
            fold_start = fold_end

        self.train_file_inds = []
        self.validation_file_inds = []
        for file_ind in self.dev_file_inds:
            if file_ind in fold_file_inds:
                self.validation_file_inds.append(file_ind)
            else:
                self.train_file_inds.append(file_ind)

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

    def make_img_generator(self, file_inds, target_size):
        for file_ind in file_inds:
            nb_rows, nb_cols = self.get_file_size(file_ind)

            if target_size is None:
                window = ((0, nb_rows), (0, nb_cols))
                img = self.get_img(file_ind, window)
                yield img, file_ind
            else:
                for row_begin in range(0, nb_rows, target_size[0]):
                    for col_begin in range(0, nb_cols, target_size[1]):
                        row_end = row_begin + target_size[0]
                        col_end = col_begin + target_size[1]
                        if row_end <= nb_rows and col_end <= nb_cols:
                            window = ((row_begin, row_end),
                                      (col_begin, col_end))
                            img = self.get_img(file_ind, window)
                            yield img, file_ind

    def make_random_img_generator(self, file_inds, target_size,
                                  sample_probs=None):
        nb_files = len(file_inds)

        while True:
            if sample_probs is None:
                rand_ind = np.random.randint(0, nb_files)
            else:
                rand_ind = np.random.choice(np.arange(0, nb_files),
                                            p=sample_probs)
            file_ind = file_inds[rand_ind]

            nb_rows, nb_cols = self.get_file_size(file_ind)

            if target_size is None:
                window = ((0, nb_rows), (0, nb_cols))
            else:
                row_begin = np.random.randint(0, nb_rows - target_size[0] + 1)
                col_begin = np.random.randint(0, nb_cols - target_size[1] + 1)
                row_end = row_begin + target_size[0]
                col_end = col_begin + target_size[1]
                window = ((row_begin, row_end), (col_begin, col_end))
            img = self.get_img(file_ind, window)

            yield img, file_ind

    def make_img_batch_generator(self, file_inds, target_size, batch_size,
                                 shuffle, sample_probs=None):
        def make_gen():
            if shuffle:
                return self.make_random_img_generator(
                    file_inds, target_size, sample_probs)
            return self.make_img_generator(file_inds, target_size)

        gen = make_gen()
        while True:
            batch, batch_file_inds = self.get_samples(gen, batch_size)
            if batch is None:
                raise StopIteration()

            yield batch, batch_file_inds

    def normalize(self, batch_x):
        means, stds = self.channel_stats
        batch_x = batch_x - means[np.newaxis, np.newaxis, np.newaxis, :]
        batch_x = batch_x / stds[np.newaxis, np.newaxis, np.newaxis, :]
        return batch_x

    def unnormalize(self, batch_x):
        means, stds = self.channel_stats
        nb_dims = len(batch_x.shape)
        if nb_dims == 3:
            batch_x = np.expand_dims(batch_x, 0)

        batch_x = batch_x * stds[np.newaxis, np.newaxis, np.newaxis, :]
        batch_x = batch_x + means[np.newaxis, np.newaxis, np.newaxis, :]

        if nb_dims == 3:
            batch_x = np.squeeze(batch_x, 0)
        return batch_x

    def augment_img_batch(self, img_batch, augment_methods):
        imgs = []
        for sample_ind in range(img_batch.shape[0]):
            img = img_batch[sample_ind, :, :, :]

            if VFLIP in augment_methods:
                if np.random.uniform() > 0.5:
                    img = np.flipud(img)

            if HFLIP in augment_methods:
                if np.random.uniform() > 0.5:
                    img = np.fliplr(img)

            if ROTATE90 in augment_methods:
                nb_rotations = np.random.randint(0, 4)
                img = np.rot90(img, nb_rotations)

            skimage_augment_methods = [ROTATE, TRANSLATE]
            if set(skimage_augment_methods).intersection(set(augment_methods)):
                # skimage requires that float images have values
                # in [-1, 1] so we have to scale and then unscale the image to
                # achieve this.
                max_val = np.max(np.absolute(img.ravel()))
                img = img / max_val
                nb_rows, nb_cols = img.shape[0:2]

                if TRANSLATE in augment_methods:
                    max_trans_ratio = 0.1
                    trans_row_bound = int(nb_rows * max_trans_ratio)
                    trans_col_bound = int(nb_cols * max_trans_ratio)
                    translation = (
                        np.random.randint(-trans_row_bound, trans_row_bound),
                        np.random.randint(-trans_col_bound, trans_col_bound)
                    )
                    tf = transform.SimilarityTransform(translation=translation)
                    img = transform.warp(img, tf, mode='reflect')

                if ROTATE in augment_methods:
                    degrees = np.random.uniform(0, 360)
                    img = transform.rotate(img, degrees, mode='reflect')

                img = img * max_val

            imgs.append(np.expand_dims(img, axis=0))

        img_batch = np.concatenate(imgs, axis=0)
        return img_batch

    def make_split_generator(self, split, target_size=None,
                             batch_size=32, shuffle=False,
                             augment_methods=None,
                             normalize=False, only_xy=True):
        file_inds = self.get_file_inds(split)
        sample_probs = None
        if split == TRAIN:
            sample_probs = self.train_probs
        elif split == VALIDATION:
            sample_probs = self.validation_probs

        img_batch_gen = self.make_img_batch_generator(
            file_inds, target_size, batch_size, shuffle, sample_probs)

        def transform_img_batch(x):
            # An img_batch is a batch of images. For segmentation problems,
            # this will contain images corresponding to both x and y. For
            # tagging problems, it will only contain images corresponding to
            # the x.
            img_batch, batch_file_inds = x
            img_batch = img_batch.astype(np.float32)

            if augment_methods:
                img_batch = self.augment_img_batch(img_batch, augment_methods)

            # The batch is an object that contains x, y, file_inds
            # and whatever else constitutes a batch for the problem type.
            batch = self.make_batch(img_batch, batch_file_inds)

            if normalize:
                batch.all_x = self.normalize(batch.all_x)
            batch.x = batch.all_x[:, :, :, self.active_input_inds]

            if only_xy:
                return batch.x, batch.y
            return batch

        split_gen = map(transform_img_batch, img_batch_gen)

        return split_gen

    def compute_channel_stats(self, batch_size, normalize):
        gen = self.make_split_generator(
            TRAIN, target_size=(10, 10), batch_size=batch_size,
            shuffle=True, augment_methods=None, normalize=normalize,
            only_xy=False)
        batch = next(gen)
        return get_channel_stats(batch.all_x)

    def write_channel_stats(self, datasets_path):
        means, stds = self.compute_channel_stats(1000, False)
        param_dict = {'means': means.tolist(), 'stds': stds.tolist()}
        param_path = join(datasets_path, self.name + '_channel_stats.json')
        save_json(param_dict, param_path)
