from os.path import join, basename, splitext, exists
import csv
import glob

import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from rastervision.common.utils import (
    save_json, compute_ndvi, plot_img_row, download_dataset, _makedirs, eprint)
from rastervision.common.data.generators import FileGenerator, Batch

PLANET_KAGGLE = 'planet_kaggle'
TIFF = 'tiff'
JPG = 'jpg'


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

        self.atmos_tags = [
            self.clear, self.cloudy, self.haze, self.partly_cloudy]
        self.common_tags = [
            self.agriculture, self.bare_ground, self.cultivation,
            self.habitation, self.primary, self.road, self.water]
        self.rare_tags = [
            self.artisinal_mine, self.blooming, self.blow_down,
            self.conventional_mine, self.selective_logging, self.slash_burn]

        self.red_ind = 0
        self.green_ind = 1
        self.blue_ind = 2
        self.rgb_inds = [self.red_ind, self.green_ind, self.blue_ind]

        self.image_shape = (256, 256)

        self.setup_channels()

    def augment_channels(self, batch_x):
        return batch_x

    def setup_channels(self):
        pass


class TiffDataset(Dataset):
    def setup_channels(self):
        self.ir_ind = 3
        self.ndvi_ind = 4
        self.nb_channels = 5

        self.display_means = np.array([0.45, 0.5, 0.5, 0.5, 0.5])
        self.display_stds = np.array([0.25, 0.2, 0.2, 0.2, 0.2])

    def augment_channels(self, batch_x):
        red = batch_x[:, :, :, [self.red_ind]]
        ir = batch_x[:, :, :, [self.ir_ind]]
        ndvi = compute_ndvi(red, ir)
        return np.concatenate([batch_x, ndvi], axis=3)


class JpgDataset(Dataset):
    def setup_channels(self):
        self.nb_channels = 3
        self.display_means = np.array([0.45, 0.5, 0.5])
        self.display_stds = np.array([0.25, 0.2, 0.2])


class TagStore():
    def __init__(self, tags_path=None, active_tags=None):
        self.dataset = Dataset()
        self.file_ind_to_tags = {}
        self.active_tags = self.dataset.all_tags if active_tags is None \
            else active_tags
        assert(set(self.active_tags) <= set(self.dataset.all_tags))

        self.tag_to_ind = dict(
            [(tag, ind) for ind, tag in enumerate(self.active_tags)])

        if tags_path is not None:
            self.load_tags(tags_path)

    def get_tag_ind(self, tag):
        if tag in self.tag_to_ind:
            return self.tag_to_ind[tag]
        return None

    def add_tags(self, file_ind, binary_tags):
        assert(binary_tags.shape[0] == len(self.active_tags))
        self.file_ind_to_tags[file_ind] = binary_tags

    def load_tags(self, tags_path):
        with open(tags_path, newline='') as tags_file:
            reader = csv.reader(tags_file)
            # Skip header
            next(reader)
            for row in reader:
                self.add_csv_row(row)

    def add_csv_row(self, row):
        file_ind, tags = row
        tags = tags.split(' ')
        self.add_tags(file_ind, self.strs_to_binary(tags))

    def strs_to_binary(self, str_tags):
        binary_tags = np.zeros((len(self.active_tags),))
        for str_tag in str_tags:
            if str_tag.strip() != '':
                ind = self.get_tag_ind(str_tag)
                if ind is not None:
                    binary_tags[ind] = 1
        return binary_tags

    def binary_to_strs(self, binary_tags):
        str_tags = []
        for tag_ind in range(len(self.active_tags)):
            if binary_tags[tag_ind] == 1:
                str_tags.append(self.active_tags[tag_ind])
        return str_tags

    def get_tag_diff(self, y_true, y_pred):
        y_true = set(self.binary_to_strs(y_true))
        y_pred = set(self.binary_to_strs(y_pred))

        add_pred_tags = sorted(list(y_pred.difference(y_true)))
        remove_pred_tags = sorted(list(y_true.difference(y_pred)))

        return add_pred_tags, remove_pred_tags

    def get_tag_array(self, file_inds):
        tags = []
        for file_ind in file_inds:
            tags.append(
                np.expand_dims(self.file_ind_to_tags[file_ind], axis=0))
        tags = np.concatenate(tags, axis=0)
        return tags

    def get_tag_counts(self):
        file_tags = self.get_tag_array(self.file_ind_to_tags.keys())
        counts = file_tags.sum(axis=0)
        tag_counts = {}
        for tag in self.active_tags:
            tag_ind = self.get_tag_ind(tag)
            tag_counts[tag] = counts[tag_ind]
        return tag_counts

    def save(self, path):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_name', 'tags'])
            for file_ind, tags in self.file_ind_to_tags.items():
                tags = ' '.join(self.binary_to_strs(tags))
                writer.writerow([file_ind, tags])

    def compute_sample_probs(self, file_inds, active_tags_prob):
        # Compute prob for each sample such that the sum of the probs of
        # samples with rare labels is equal to active_tags_prob
        tag_array = self.get_tag_array(file_inds)
        active_tag_file_inds = []
        for tag in self.active_tags:
            tag_ind = self.get_tag_ind(tag)
            if tag_ind is not None:
                active_tag_file_inds.append(tag_ind)

        is_active_tag_file_ind = np.any(
            tag_array[:, active_tag_file_inds], axis=1)
        nb_active_tag_file_inds = np.sum(is_active_tag_file_ind)
        nb_other_file_inds = len(file_inds) - nb_active_tag_file_inds

        if nb_active_tag_file_inds == 0:
            active_tags_prob = 0.
        if nb_other_file_inds == 0:
            active_tags_prob = 1.

        other_per_sample_prob = 0.
        active_tag_per_sample_prob = 0.
        if nb_other_file_inds > 0:
            other_per_sample_prob = \
                (1.0 - active_tags_prob) / nb_other_file_inds
        if nb_active_tag_file_inds > 0:
            active_tag_per_sample_prob = \
                active_tags_prob / nb_active_tag_file_inds

        sample_probs = np.ones((len(file_inds),)) * other_per_sample_prob
        sample_probs[is_active_tag_file_ind] = active_tag_per_sample_prob

        return sample_probs


class PlanetKaggleFileGenerator(FileGenerator):
    def __init__(self, datasets_path, options):
        download_dataset(PLANET_KAGGLE, self.file_names)

        self.dataset_path = join(datasets_path, PLANET_KAGGLE)
        self.dev_path = join(self.dataset_path, self.dev_dir)
        self.test_path = join(self.dataset_path, self.test_dir)

        self.drop_file_inds = []
        if self.drop_file_inds_file:
            with open(join(self.dataset_path, self.drop_file_inds_file)) as f:
                self.drop_file_inds = f.read().split('\n')

        self.dev_file_inds = self.generate_file_inds(self.dev_path)
        self.test_file_inds = self.generate_file_inds(self.test_path)

        self.active_tags = options.active_tags
        self.active_tags = options.active_tags \
            if options.active_tags is not None else self.dataset.all_tags
        self.active_tags_prob = options.active_tags_prob

        tags_path = join(self.dataset_path, 'train_v2.csv')
        self.tag_store = TagStore(
            tags_path=tags_path, active_tags=self.active_tags)

        super().__init__(options)

    def compute_train_probs(self):
        if self.active_tags_prob is not None:
            return self.tag_store.compute_sample_probs(
                self.train_file_inds, self.active_tags_prob)
        return None

    @staticmethod
    def preprocess(datasets_path):
        dataset_path = join(datasets_path, PLANET_KAGGLE)
        tags_path = join(dataset_path, 'train_v2.csv')
        tag_store = TagStore(tags_path=tags_path)
        counts_path = join(dataset_path, 'tag_counts.json')
        save_json(tag_store.get_tag_counts(), counts_path)

    def generate_file_inds(self, path):
        file_inds = []
        if isinstance(self.file_extension, list):
            for fe in self.file_extension:
                paths = sorted(glob.glob(join(path, '*.{}'.format(fe))))
                for path in paths:
                    file_ind = splitext(basename(path))[0]
                    if not file_ind in self.drop_file_inds:
                        file_inds.append(file_ind)
        else:
            paths = sorted(glob.glob(join(path, '*.{}'.format(self.file_extension))))
            for path in paths:
                file_ind = splitext(basename(path))[0]
                if not file_ind in self.drop_file_inds:
                    file_inds.append(file_ind)

        eprint("FILE INDS SIZE: %d" % len(file_inds))
        return file_inds

    def plot_sample(self, file_path, x, y, file_ind):
        fig = plt.figure()
        nb_cols = self.dataset.nb_channels + 1
        grid_spec = mpl.gridspec.GridSpec(1, nb_cols)

        # Plot x channels
        x = self.calibrate_image(x)
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
        prefix, _ = file_ind.split('_')
        data_dir = self.test_path if prefix in ['file', 'test'] \
            else self.dev_path
        if isinstance(self.file_extension, list):
            for fe in self.file_extension:
                n = '{}.{}'.format(file_ind, fe)
                p = join(self.dataset_path, data_dir, n)
                if exists(p):
                    return p
            raise IOError("File %s.{%s} does not exist" % (file_ind, '|'.join(self.file_extension)))
        else:
            n = '{}.{}'.format(file_ind, self.file_extension)
            return join(self.dataset_path, data_dir, n)

    def get_file_size(self, file_ind):
        return 256, 256

    def get_img(self, file_ind, window):
        file_path = self.get_file_path(file_ind)
        img = self.load_img(file_path, window)
        return img

    def make_batch(self, img_batch, file_inds):
        batch = Batch()
        batch.all_x = self.dataset.augment_channels(img_batch)
        batch.file_inds = file_inds

        if self.has_y(file_inds[0]):
            batch.y = self.tag_store.get_tag_array(file_inds)
        return batch


class PlanetKaggleTiffFileGenerator(PlanetKaggleFileGenerator):
    def __init__(self, datasets_path, options):
        self.dev_dir = 'train-tif-v2'
        self.test_dir = 'test-tif-v2' # Zip is named v3, but unzipped still named v2
        self.file_names = [
            'train-tif-v2.zip', 'test-tif-v3.zip', 'train_v2.csv.zip',
            'planet_kaggle_tiff_channel_stats.json', 'unaligned_tifs.csv']
        self.file_extension = 'tif'
        self.dataset = TiffDataset()
        self.name = 'planet_kaggle_tiff'
        self.drop_file_inds_file = 'unaligned_tifs.csv'

        super().__init__(datasets_path, options)

    def load_img(self, file_path, window):
        import rasterio
        with rasterio.open(file_path) as src:
            b, g, r, nir = src.read(window=window)
            img = np.dstack([r, g, b, nir])
            return img

    @staticmethod
    def preprocess(datasets_path):
        PlanetKaggleFileGenerator.preprocess(datasets_path)

        proc_data_path = join(datasets_path, PLANET_KAGGLE)
        _makedirs(proc_data_path)

        class Options():
            def __init__(self):
                self.active_input_inds = [0, 1, 2, 3]
                self.train_ratio = 0.8
                self.cross_validation = None
                self.active_tags_prob = None
                self.active_tags = None

        options = Options()
        PlanetKaggleTiffFileGenerator(
            datasets_path, options).write_channel_stats(proc_data_path)


class PlanetKaggleJpgFileGenerator(PlanetKaggleFileGenerator):
    def __init__(self, datasets_path, options):
        self.dev_dir = 'train-jpg'
        self.test_dir = 'test-jpg'
        self.file_names = [
            'train-jpg.zip', 'test-jpg.zip', 'train_v2.csv.zip',
            'planet_kaggle_jpg_channel_stats.json']
        self.file_extension = 'jpg'
        self.dataset = JpgDataset()
        self.name = 'planet_kaggle_jpg'
        self.drop_file_inds_file = None

        super().__init__(datasets_path, options)

    def load_img(self, file_path, window):
        import rasterio
        if not exists(file_path):
            raise IOError("Image does not exist: %s" % file_path)
        with rasterio.open(file_path) as src:
            r, g, b = src.read(window=window)
            img = np.dstack([r, g, b])
            return img

    @staticmethod
    def preprocess(datasets_path):
        PlanetKaggleFileGenerator.preprocess(datasets_path)

        proc_data_path = join(datasets_path, PLANET_KAGGLE)
        _makedirs(proc_data_path)

        class Options():
            def __init__(self):
                self.active_input_inds = [0, 1, 2]
                self.train_ratio = 0.8
                self.cross_validation = None
                self.active_tags_prob = None
                self.active_tags = None

        options = Options()
        PlanetKaggleJpgFileGenerator(
            datasets_path, options).write_channel_stats(proc_data_path)
