import numpy as np

from rastervision.core.ml_task import MLTask


class SemanticSegmentation(MLTask):
    def get_train_windows(self, scene, options):
        raster_source = scene.raster_source
        extent = raster_source.get_extent()

        # label_store = scene.ground_truth_label_store

        chip_size = options.chip_size

        windows = []
        for i in range(100):  # XXX insensitive
            windows.append(extent.make_random_square(chip_size))

        return windows

    def get_train_labels(self, window, scene, options):
        label_store = scene.ground_truth_label_store
        chip = label_store.src._get_chip(window)
        fn = label_store.fn

        bit2 = (chip[:, :, 0] > 0)
        bit1 = (chip[:, :, 1] > 0)
        bit0 = (chip[:, :, 2] > 0)
        retval = np.array(bit2 * 4 + bit1 * 2 + bit0 * 1, dtype=np.uint8)

        return np.array(fn(retval), dtype=np.uint8)
