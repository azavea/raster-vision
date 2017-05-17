from os.path import join, basename, splitext, isdir, dirname
import csv
import glob
from subprocess import call

import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from rastervision.common.utils import (
    save_json, compute_ndvi, plot_img_row, download_dataset)
from rastervision.common.generators import FileGenerator, Batch

PLANET_KAGGLE = 'planet_kaggle'
TIFF = 'tiff'

DEV_DIR = 'train-tif-v2'
TEST_DIR = 'test-tif-v2'


class Dataset():
    def __init__(self):
        self.agriculture = 'agriculture'
        self.artisinal_mine = 'artisinal_mine'
        self.bare_ground = 'bare_ground'
        self.blooming = 'blooming'
        self.blow_down = 'blow_down'
        self.clear = 'clear'
        self.cloudy = 'cloudy'
        self.conventional_mine = 'conventional_mine'
        self.cultivation = 'cultivation'
        self.habitation = 'habitation'
        self.haze = 'haze'
        self.partly_cloudy = 'partly_cloudy'
        self.primary = 'primary'
        self.road = 'road'
        self.selective_logging = 'selective_logging'
        self.slash_burn = 'slash_burn'
        self.water = 'water'

        self.all_tags = [
            self.agriculture, self.artisinal_mine, self.bare_ground,
            self.blooming, self.blow_down, self.clear, self.cloudy,
            self.conventional_mine, self.cultivation, self.habitation,
            self.haze, self.partly_cloudy, self.primary, self.road,
            self.selective_logging, self.slash_burn, self.water]
        self.nb_tags = len(self.all_tags)
        self.tag_to_ind = dict(
            [(tag, ind) for ind, tag in enumerate(self.all_tags)])

        self.atmos_tags = [
            self.clear, self.cloudy, self.haze, self.partly_cloudy]
        self.common_tags = [
            self.agriculture, self.bare_ground, self.cultivation,
            self.habitation, self.primary, self.road, self.water]
        self.rare_tags = [
            self.artisinal_mine, self.blooming, self.blow_down, self.blooming,
            self.conventional_mine, self.selective_logging, self.slash_burn]

        self.red_ind = 0
        self.green_ind = 1
        self.blue_ind = 2
        self.rgb_inds = [self.red_ind, self.green_ind, self.blue_ind]
        self.ir_ind = 3
        self.ndvi_ind = 4
        self.nb_channels = 5

        self.image_shape = (256, 256)

    def augment_channels(self, batch_x):
        red = batch_x[:, :, :, [self.red_ind]]
        ir = batch_x[:, :, :, [self.ir_ind]]
        ndvi = compute_ndvi(red, ir)
        return np.concatenate([batch_x, ndvi], axis=3)


class TagStore():
    def __init__(self, tags_path, dataset):
        self.dataset = dataset
        self.load_tags(tags_path)

    def load_tags(self, tags_path):
        self.file_ind_to_tags = {}
        with open(tags_path, newline='') as tags_file:
            reader = csv.reader(tags_file)
            # Skip header
            next(reader)

            for line_ind, row in enumerate(reader):
                file_ind, tags = row
                tags = tags.split(' ')
                self.file_ind_to_tags[file_ind] = self.strs_to_binary(tags)

    def strs_to_binary(self, str_tags):
        binary_tags = np.zeros((self.dataset.nb_tags,))
        for str_tag in str_tags:
            ind = self.dataset.tag_to_ind[str_tag]
            binary_tags[ind] = 1
        return binary_tags

    def binary_to_strs(self, binary_tags):
        str_tags = []
        for tag_ind in range(self.dataset.nb_tags):
            if binary_tags[tag_ind] == 1:
                str_tags.append(self.dataset.all_tags[tag_ind])
        return str_tags

    def get_tags(self, file_inds):
        tags = []
        for file_ind in file_inds:
            tags.append(
                np.expand_dims(self.file_ind_to_tags[file_ind], axis=0))
        tags = np.concatenate(tags, axis=0)
        return tags

    def get_tag_counts(self, tags):
        file_tags = self.get_tags(self.file_ind_to_tags.keys())
        counts = file_tags.sum(axis=0)
        tag_counts = {}
        for tag in tags:
            tag_ind = self.tag_to_ind[tag]
            tag_counts[tag] = counts[tag_ind]
        return tag_counts


class PlanetKaggleFileGenerator(FileGenerator):
    def __init__(self, active_input_inds, train_ratio, cross_validation):
        tags_path = join(self.dataset_path, 'train_v2.csv')
        self.dataset = Dataset()
        self.tag_store = TagStore(tags_path, self.dataset)

        super().__init__(active_input_inds, train_ratio, cross_validation)

    @staticmethod
    def preprocess(datasets_path):
        dataset_path = join(datasets_path, PLANET_KAGGLE)
        tags_path = join(dataset_path, 'train_v2.csv')
        dataset = Dataset()
        tag_store = TagStore(tags_path, dataset)
        counts_path = join(dataset_path, 'tag_counts.json')
        save_json(tag_store.get_tag_counts(dataset.all_tags), counts_path)


class PlanetKaggleTiffFileGenerator(PlanetKaggleFileGenerator):
    def __init__(self, datasets_path, active_input_inds, train_ratio,
                 cross_validation):
        self.download_dataset()

        self.dataset_path = join(datasets_path, PLANET_KAGGLE)
        self.dev_path = join(self.dataset_path, DEV_DIR)
        self.test_path = join(self.dataset_path, TEST_DIR)

        self.dev_file_inds = self.generate_file_inds(self.dev_path)
        self.test_file_inds = self.generate_file_inds(self.test_path)

        super().__init__(active_input_inds, train_ratio, cross_validation)

    def download_dataset(self):
        file_names = [
            'train-tif-v2.zip', 'test-tif-v2.zip', 'train_v2.csv.zip']
        download_dataset(PLANET_KAGGLE, file_names)

    def generate_file_inds(self, path):
        paths = glob.glob(join(path, '*.tif'))
        file_inds = []
        for path in paths:
            file_ind = splitext(basename(path))[0]
            file_inds.append(file_ind)
        return file_inds

    def display_scale(self, x):
        # TODO do this in a better way following the kaggle jupiter notebook
        x = np.copy(x)
        rgbir_x = x[:, :, 0:4]
        ndvi_x = x[:, :, 4]
        uint16_max = float(np.iinfo(np.uint16).max)

        rgbir_x /= uint16_max
        rgbir_x *= 3

        ndvi_x += 1.0
        ndvi_x /= 2

        return x

    def plot_sample(self, file_path, x, y, file_ind):
        fig = plt.figure()
        nb_cols = self.dataset.nb_channels + 1
        grid_spec = mpl.gridspec.GridSpec(1, nb_cols)

        # Plot x channels
        x = self.unnormalize(x)
        x = self.display_scale(x)
        rgb_x = x[:, :, self.dataset.rgb_inds]
        imgs = [rgb_x]
        nb_channels = x.shape[2]
        for channel_ind in range(nb_channels):
            img = x[:, :, channel_ind]
            imgs.append(img)
        row_ind = 0
        plot_img_row(fig, grid_spec, row_ind, imgs)

        # Print y tags
        tag_strs = self.tag_store.binary_to_strs(y)
        tag_strs = ', '.join(tag_strs)
        title = 'file_ind:{} tags: {}'.format(file_ind, tag_strs)
        fig.suptitle(title)

        plt.savefig(file_path, bbox_inches='tight', format='pdf', dpi=600)
        plt.close(fig)

    def get_file_path(self, file_ind):
        split, _ = file_ind.split('_')
        data_dir = self.test_path if split == 'test' else self.dev_path
        return join(
            self.dataset_path, data_dir, '{}.tif'.format(file_ind))

    def get_file_size(self, file_ind):
        return 256, 256

    def load_img(self, file_path, window):
        import rasterio
        with rasterio.open(file_path) as src:
            b, g, r, nir = src.read(window=window)
            img = np.dstack([r, g, b, nir])
            return img

    def get_img(self, file_ind, window):
        file_path = self.get_file_path(file_ind)
        img = self.load_img(file_path, window)
        return img

    def make_batch(self, img_batch, file_inds):
        batch = Batch()
        batch.all_x = img_batch
        batch.all_x = self.dataset.augment_channels(batch.all_x)
        batch.file_inds = file_inds

        if self.has_y(file_inds[0]):
            batch.y = self.tag_store.get_tags(file_inds)
        return batch
