from os.path import join

import numpy as np

from .isprs import IsprsDataset
from .generators import FileGenerator, TRAIN, VALIDATION, TEST
from .utils import (
    load_image, get_image_size, compute_ndvi, _makedirs,
    save_numpy_array)

VAIHINGEN = 'vaihingen'
PROCESSED_VAIHINGEN = 'processed_vaihingen'


class VaihingenDataset(IsprsDataset):
    def __init__(self, include_depth=False, include_ndvi=False):
        self.include_ir = True
        self.include_depth = include_depth
        self.include_ndvi = include_ndvi
        self.setup_channels()
        super().__init__()

    def setup_channels(self):
        self.ir_ind = 0
        self.red_ind = 1
        self.green_ind = 2
        self.irrg_input_inds = [self.ir_ind, self.red_ind, self.green_ind]
        self.rgb_input_inds = self.irrg_input_inds

        curr_ind = 2

        if self.include_depth:
            curr_ind += 1
            self.depth_ind = curr_ind

        if self.include_ndvi:
            curr_ind += 1
            self.ndvi_ind = curr_ind

        self.nb_channels = curr_ind + 1

    def get_output_file_name(self, file_ind):
        return 'top_mosaic_09cm_area{}.tif'.format(file_ind)


class VaihingenFileGenerator(FileGenerator):
    """
    A data generator for the Vaihingen dataset that creates batches from
    files on disk.
    """
    def __init__(self, include_depth, include_ndvi, train_ratio):
        self.dataset = VaihingenDataset(include_depth, include_ndvi)
        self.train_ratio = train_ratio

        self.file_inds = [
            1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]

        self.test_file_inds = [
            2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]

        super().__init__()


class VaihingenImageFileGenerator(VaihingenFileGenerator):
    """
    A data generator for the Vaihingen dataset that creates batches from
    the original TIFF and JPG files.
    """
    def __init__(self, datasets_path, include_depth=False, include_ndvi=False,
                 train_ratio=0.8):
        self.dataset_path = join(datasets_path, VAIHINGEN)
        super().__init__(include_depth, include_ndvi, train_ratio)

    @staticmethod
    def preprocess(datasets_path):
        pass

    def get_file_size(self, file_ind):
        irrg_file_path = join(
            self.dataset_path,
            'top/top_mosaic_09cm_area{}.tif'.format(file_ind))
        nb_rows, nb_cols = get_image_size(irrg_file_path)
        return nb_rows, nb_cols

    def get_tile(self, file_ind, window, has_outputs=True):
        irrg_file_path = join(
            self.dataset_path,
            'top/top_mosaic_09cm_area{}.tif'.format(file_ind))
        depth_file_path = join(
            self.dataset_path,
            'dsm/dsm_09cm_matching_area{}.tif'.format(file_ind))
        outputs_file_path = join(
            self.dataset_path,
            'gts_for_participants/top_mosaic_09cm_area{}.tif'.format(file_ind))
        outputs_no_boundary_file_path = join(
            self.dataset_path,
            'ISPRS_semantic_labeing_Vaihingen_ground_truth_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'.format(file_ind)) # noqa

        irrg = load_image(irrg_file_path, window)
        depth = load_image(depth_file_path, window)
        depth = ((depth - 240) * 2).astype(np.uint8)
        channels = [irrg, depth]

        if has_outputs:
            outputs = load_image(outputs_file_path, window)
            outputs_no_boundary = load_image(
                outputs_no_boundary_file_path, window)
            channels.extend([outputs, outputs_no_boundary])

        tile = np.concatenate(channels, axis=2)
        return tile

    def parse_batch(self, batch, has_outputs=True):
        irrg = batch[:, :, :, 0:3]
        depth = batch[:, :, :, 3:4]

        input_channels = [irrg]
        if self.dataset.include_depth:
            input_channels.append(depth)
        if self.dataset.include_ndvi:
            ir = irrg[:, :, :, 0:1]
            red = irrg[:, :, :, 1:2]
            ndvi = compute_ndvi(red, ir)
            input_channels.append(ndvi)

        inputs = np.concatenate(input_channels, axis=3)

        outputs = None
        outputs_mask = None
        if has_outputs:
            outputs = self.dataset.rgb_to_one_hot_batch(batch[:, :, :, 4:7])
            outputs_mask = self.dataset.rgb_to_mask_batch(batch[:, :, :, 7:])
        return inputs, outputs, outputs_mask


class VaihingenNumpyFileGenerator(VaihingenFileGenerator):
    """
    A data generator for the Vaihingen dataset that creates batches from
    numpy array files. This is about 20x faster than reading the raw files.
    """
    def __init__(self, datasets_path, include_depth=False,
                 include_ndvi=False, train_ratio=0.8):
        self.raw_dataset_path = join(datasets_path, VAIHINGEN)
        self.dataset_path = join(datasets_path, PROCESSED_VAIHINGEN)
        super().__init__(include_depth, include_ndvi, train_ratio)

    @staticmethod
    def preprocess(datasets_path):
        proc_data_path = join(datasets_path, PROCESSED_VAIHINGEN)
        _makedirs(proc_data_path)

        generator = VaihingenImageFileGenerator(
            datasets_path, include_depth=True,
            include_ndvi=False)
        dataset = generator.dataset

        def _preprocess(split):
            gen = generator.make_split_generator(
                split, batch_size=1, shuffle=False, augment=False,
                normalize=False, eval_mode=True)

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

                file_name = '{}'.format(file_ind)
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
        return join(self.dataset_path, '{}.npy'.format(file_ind))

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
        irrg = batch[:, :, :, 0:3]
        depth = batch[:, :, :, 3:4]

        input_channels = [irrg]
        if self.dataset.include_depth:
            input_channels.append(depth)
        if self.dataset.include_ndvi:
            ir = irrg[:, :, :, 0:1]
            red = irrg[:, :, :, 1:2]
            ndvi = compute_ndvi(red, ir)
            input_channels.append(ndvi)

        inputs = np.concatenate(input_channels, axis=3)
        outputs = None
        outputs_mask = None
        if has_outputs:
            outputs = self.dataset.label_to_one_hot_batch(batch[:, :, :, 4:5])
            outputs_mask = batch[:, :, :, 5:6]
        return inputs, outputs, outputs_mask
