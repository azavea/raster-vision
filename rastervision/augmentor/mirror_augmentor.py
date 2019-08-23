import random
import numpy as np
from rastervision.core import TrainingData
from rastervision.augmentor import Augmentor
from rastervision.core import TrainingData


class MirrorAugmentor(Augmentor):
	"""Randomly add a mirrored copy of the chips to the data.

	In case where there is not much training data, this might
	increase the amount of training data. It is meant to be used with
	the chip-classification task, as the labels are not mirrored. This would
	result in wrongly placed labels for any other task than chip-classification.


	Args:
		prob: (float) probability that the data is mirrord

	"""

	def __init__(self, aug_prob):
		self.aug_prob = aug_prob

	def process(self, training_data, tmp_dir):
		augmented_data = TrainingData() # Initiatlize empty training data object

		# Define functions that do the mirroring
		def mirror_horizontally(chip):
			return np.flip(np.copy(chip), axes = 1)

		def mirror_vertically(chip):
			return np.copy(chip, axes = 0)

		def mirror_diagonal1(chip):
			return np.transpose(np.copy(chip), axes = [1,0,2])

		def mirror_diagonal2(chip):
			return np.transpose(np.copy(mirror_horizontally(chip)), axes = [1,0,2])

		for chip, window, labels in training_data:
			if random.uniform(0,1) < self.aug_prob:
				chip = np.copy(chip)
				augmented_data.append(mirror_horizontally(chip), window, labels)
				augmented_data.append(mirror_vertically(chip), window, labels)
				augmented_data.append(mirror_diagonal1(chip), window, labels)
				augmented_data.append(mirror_diagonal2(chip), window, labels)

		return augmented_data
