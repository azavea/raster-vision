import random

import numpy as np

from rastervision.augmentor import Augmentor
from rastervision.core import (TrainingData, Box)


class NodataAugmentor(Augmentor):
    """Randomly add NoData values to negative chips.

    This is useful for training the model to negatively predict
    chips that are on boundaries or containing mostly NoData.
    """

    def __init__(self, aug_prob):
        self.aug_prob = aug_prob

    def process(self, training_data, tmp_dir):
        augmented = TrainingData()
        nodata_aug_prob = self.aug_prob

        for chip, window, labels in training_data:
            # If negative chip, with some probability, add a random black square
            # to chip.
            if len(labels) == 0 and random.uniform(0, 1) < nodata_aug_prob:
                size = round(random.uniform(0, 1) * chip.shape[0])
                square = Box(0, 0, chip.shape[0],
                             chip.shape[1]).make_random_square(size)
                chip = np.copy(chip)
                chip[square.ymin:square.ymax, square.xmin:square.xmax, :] = 0

            augmented.append(chip, window, labels)

        return augmented
