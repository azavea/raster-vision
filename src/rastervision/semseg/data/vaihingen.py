from os.path import join

import numpy as np

from rastervision.common.utils import (
    load_img, get_img_size, _makedirs,
    save_numpy_array)
from rastervision.common.settings import (
    TRAIN, VALIDATION, TEST
)

from rastervision.semseg.data.isprs import (
    IsprsDataset, IsprsBatch, IsprsFileGenerator)

VAIHINGEN = 'isprs/vaihingen'
PROCESSED_VAIHINGEN = 'isprs/processed_vaihingen'


class VaihingenDataset(IsprsDataset):
    def __init__(self):
        self.ir_ind = 0
        self.red_ind = 1
        self.green_ind = 2
        self.rgb_inds = [self.ir_ind, self.red_ind, self.green_ind]

        self.depth_ind = 3
        self.ndvi_ind = 4

        self.nb_channels = 5

        self.display_means = np.array([0.5] * self.nb_channels)
        self.display_stds = np.array([0.2] * self.nb_channels)
        self.display_means[self.green_ind] = 0.45

        super().__init__()

    def get_output_file_name(self, file_ind):
        return 'top_mosaic_09cm_area{}.tif'.format(file_ind)


class VaihingenFileGenerator(IsprsFileGenerator):
    """
    A data generator for the Vaihingen dataset that creates batches from
    files on disk.
    """
    def __init__(self, options):
        self.dataset = VaihingenDataset()

        self.dev_file_inds = [
            1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]

        self.test_file_inds = [
            2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]

        super().__init__(options)


class VaihingenImageFileGenerator(VaihingenFileGenerator):
    """
    A data generator for the Vaihingen dataset that creates batches from
    the original TIFF and JPG files.
    """
    def __init__(self, datasets_path, options):
        self.dataset_path = join(datasets_path, VAIHINGEN)
        self.name = "vaihingen_image"
        super().__init__(options)

    @staticmethod
    def preprocess(datasets_path):
        proc_data_path = join(datasets_path, PROCESSED_VAIHINGEN)
        _makedirs(proc_data_path)

        class Options():
            def __init__(self):
                self.active_input_inds = [0, 1, 2, 3]
                self.train_ratio = 0.8
                self.cross_validation = None
        options = Options()
        VaihingenImageFileGenerator(
            datasets_path, options).write_channel_stats(proc_data_path)

    def get_file_size(self, file_ind):
        irrg_file_path = join(
            self.dataset_path,
            'top/top_mosaic_09cm_area{}.tif'.format(file_ind))
        nb_rows, nb_cols = get_img_size(irrg_file_path)
        return nb_rows, nb_cols

    def get_img(self, file_ind, window):
        irrg_file_path = join(
            self.dataset_path,
            'top/top_mosaic_09cm_area{}.tif'.format(file_ind))
        depth_file_path = join(
            self.dataset_path,
            'dsm/dsm_09cm_matching_area{}.tif'.format(file_ind))
        batch_y_file_path = join(
            self.dataset_path,
            'gts_for_participants/top_mosaic_09cm_area{}.tif'.format(file_ind))
        batch_y_no_boundary_file_path = join(
            self.dataset_path,
            'ISPRS_semantic_labeing_Vaihingen_ground_truth_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'.format(file_ind)) # noqa

        irrg = load_img(irrg_file_path, window)
        depth = load_img(depth_file_path, window)
        depth = ((depth - 240) * 2).astype(np.uint8)
        channels = [irrg, depth]

        if self.has_y(file_ind):
            batch_y = load_img(batch_y_file_path, window)
            batch_y_no_boundary = load_img(
                batch_y_no_boundary_file_path, window)
            channels.extend([batch_y, batch_y_no_boundary])

        img = np.concatenate(channels, axis=2)
        return img

    def make_batch(self, img_batch, file_inds):
        batch = IsprsBatch()
        batch.all_x = img_batch[:, :, :, 0:4]
        batch.all_x = self.dataset.augment_channels(batch.all_x)
        batch.file_inds = file_inds

        if self.has_y(file_inds[0]):
            batch.y = self.dataset.rgb_to_one_hot_batch(
                img_batch[:, :, :, 4:7])
            batch.y_mask = self.dataset.rgb_to_mask_batch(
                img_batch[:, :, :, 7:])
        return batch


class VaihingenNumpyFileGenerator(VaihingenFileGenerator):
    """
    A data generator for the Vaihingen dataset that creates batches from
    numpy array files. This is about 20x faster than reading the raw files.
    """
    def __init__(self, datasets_path, options):
        self.raw_dataset_path = join(datasets_path, VAIHINGEN)
        self.dataset_path = join(datasets_path, PROCESSED_VAIHINGEN)
        self.download_dataset(['processed_vaihingen.zip'])
        self.name = "vaihingen_numpy"
        super().__init__(options)

    @staticmethod
    def preprocess(datasets_path):
        proc_data_path = join(datasets_path, PROCESSED_VAIHINGEN)
        _makedirs(proc_data_path)

        class Options():
            def __init__(self):
                self.active_input_inds = [0, 1, 2, 3]
                self.train_ratio = 0.8
                self.cross_validation = None
        options = Options()
        generator = VaihingenImageFileGenerator(datasets_path, options)
        dataset = generator.dataset

        def _preprocess(split):
            gen = generator.make_split_generator(
                split, batch_size=1, shuffle=False, augment_methods=None,
                normalize=False, only_xy=False)

            for batch in gen:
                print('.')
                file_ind = batch.file_inds[0]
                x = np.squeeze(batch.x, axis=0)
                channels = [x]

                if batch.y is not None:
                    y = np.squeeze(batch.y, axis=0)
                    y = dataset.one_hot_to_label_batch(y)
                    y_mask = np.squeeze(batch.y_mask, axis=0)
                    channels.extend([y, y_mask])
                channels = np.concatenate(channels, axis=2)

                file_name = '{}'.format(file_ind)
                save_numpy_array(
                    join(proc_data_path, file_name), channels)

                # Free memory
                channels = None
                batch.all_x = None
                batch.x = x = None
                batch.y = y = None
                batch.y_mask = y_mask = None

        _preprocess(TRAIN)
        _preprocess(VALIDATION)
        _preprocess(TEST)

        VaihingenNumpyFileGenerator(
            datasets_path, options).write_channel_stats(proc_data_path)

    def get_file_path(self, file_ind):
        return join(self.dataset_path, '{}.npy'.format(file_ind))

    def get_file_size(self, file_ind):
        file_path = self.get_file_path(file_ind)
        im = np.load(file_path, mmap_mode='r')
        nb_rows, nb_cols = im.shape[0:2]
        return nb_rows, nb_cols

    def get_img(self, file_ind, window):
        file_path = self.get_file_path(file_ind)
        im = np.load(file_path, mmap_mode='r')
        ((row_begin, row_end), (col_begin, col_end)) = window
        img = im[row_begin:row_end, col_begin:col_end, :]

        return img

    def make_batch(self, img_batch, file_inds):
        batch = IsprsBatch()
        batch.all_x = img_batch[:, :, :, 0:4]
        batch.all_x = self.dataset.augment_channels(batch.all_x)
        batch.file_inds = file_inds

        if self.has_y(file_inds[0]):
            batch.y = self.dataset.label_to_one_hot_batch(
                img_batch[:, :, :, 4:5])
            batch.y_mask = img_batch[:, :, :, 5:6]
        return batch
