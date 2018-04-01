import numpy as np

from rastervision.core.ml_task import MLTask
from rastervision.evaluations.classification_evaluation import (
    ClassificationEvaluation)


class Classification(MLTask):
    def get_train_windows(self, project, options):
        extent = project.raster_source.get_extent()
        chip_size = options.chip_size
        stride = chip_size
        windows = []
        for window in extent.get_windows(chip_size, stride):
            chip = project.raster_source.get_chip(window)
            if np.sum(chip.ravel()) > 0:
                windows.append(window)
        return windows

    def get_train_labels(self, window, project, options):
        return project.ground_truth_label_store.get_labels(window)

    def get_predict_windows(self, extent, options):
        chip_size = options.chip_size
        stride = chip_size
        return extent.get_windows(chip_size, stride)

    def get_evaluation(self):
        return ClassificationEvaluation()
