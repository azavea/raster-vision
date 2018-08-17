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
                  `make_chips` section of the workflow
                  configuration file.

        Returns:
             A list of windows, list(Box)

        """
        seg_options = options.segmentation_options
        raster_source = scene.raster_source
        extent = raster_source.get_extent()
        label_store = scene.ground_truth_label_store
        chip_size = options.chip_size
        p = seg_options.empty_survival_probability

        windows = []
        while (len(windows) < seg_options.number_of_chips):
            window = extent.make_random_square(chip_size)
            if label_store.has_labels(window):
                windows.append(window)
            elif (p > 0) and (np.random.rand() <= p):
                windows.append(window)

        return windows

    def get_train_labels(self, window: Box, scene: Scene,
                         options) -> np.ndarray:
        """Get the training labels for the given window in the given scene.

        Args:
             window: The window over-which the labels are to be
                  retrieved.
             scene: The scene from-which the window of labels is to be
                  extracted.
             options: Options passed through from the
                  `make_chips` section of the workflow
                  configuration file.

        Returns:
             An appropriately-shaped 2d np.ndarray with the labels
             encoded as packed pixels.

        """
        label_store = scene.ground_truth_label_store
        return label_store.get_labels(window)
