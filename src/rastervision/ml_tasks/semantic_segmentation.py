import numpy as np

from typing import List

from rastervision.core.box import Box
from rastervision.core.ml_task import MLTask
from rastervision.core.scene import Scene


class SemanticSegmentation(MLTask):
    """MLTask-derived type that implements the semantic segmentation task.

    """

    def get_train_windows(self, scene: Scene, options) -> List[Box]:
        """Get training windows covering a scene.

        Args:
             scene: The scene over-which windows are to be generated.
             options: Options passed through from the
                  `make_training_chips` section of the workflow
                  configuration file.

        Return:
             A list of windows, list(Box)

        """
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
