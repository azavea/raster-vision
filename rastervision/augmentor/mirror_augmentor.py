import random
import numpy as np
from rastervision.core import TrainingData

class MirrorAugmentor(Augmentor):
	"""Randomly add a mirrored copy of the chips to the data.

	In case where there is not much training data, this might increase the amount of training data

	Args:
		prob: probability that the data is mirrord

		axes: (int) Either 4 or 8. When 4 is used, the images are mirrored along their horizontal
		and vertical axes. If 8 is used, also the diagonal axes (transpose and transverse) are used.
	"""

	def __init__(self, aug_prob, axes):
		self.aug_prob = aug_prob
		self.axes = axes

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

		# If loop on top level to avoid a per-chip if statement
		if self.axes == 4:
			for chip, window, labels in training_data:
				if random.uniform(0,1) < self.aug_prob:
					chip = np.copy(chip)
					augmented_data.append(mirror_horizontally(chip), window, labels)
					augmented_data.append(mirror_vertically(chip), window, labels)

		elif self.axes == 8:
			for chip, window, labels in training_data:
				if random.uniform(0,1) < self.aug_prob:
					chip = np.copy(chip)
					augmented_data.append(mirror_horizontally(chip), window, labels)
					augmented_data.append(mirror_vertically(chip), window, labels)
					augmented_data.append(mirror_diagonal1(chip), window, labels)
					augmented_data.append(mirror_diagonal2(chip), window, labels)

		return augmented_data
