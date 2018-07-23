import numpy as np

from rastervision.core.ml_task import MLTask
from rastervision.core.box import Box


class SemanticSegmentation(MLTask):
    def get_train_windows(self, scene, options):
        raster_source = scene.raster_source
        meta = raster_source.image_dataset.meta
        width = meta.get('width')
        height = meta.get('height')

        # label_store = scene.ground_truth_label_store

        chip_size = options.chip_size

        windows = []
        for i in range(100):  # XXX insensitive
            xmin = np.random.randint(low=0, high=width - chip_size)
            ymin = np.random.randint(low=0, high=height - chip_size)
            windows.append(Box.make_square(ymin, xmin, chip_size))

        return windows

    def get_train_labels(self, window, scene, options):
        chip = scene.ground_truth_label_store.src.get_chip(window)
        bit2 = (chip[:, :, 0] > 0)
        bit1 = (chip[:, :, 1] > 0)
        bit0 = (chip[:, :, 2] > 0)
        return np.array(bit2 * 4 + bit1 * 2 + bit0 * 1, dtype=np.uint8)
