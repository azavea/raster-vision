import random
import numpy as np
from rastervision.core import TrainingData
from rastervision.augmentor import Augmentor
from rastervision.core import TrainingData
import scipy.misc # For testing, saves the images


class MirrorAugmentor(Augmentor):
	"""Add mirrored copies of the chips to the data.

	In case where there is not much training data, this can be used to
	increase the amount of training data. Data is mirrored along 4 axes.
	First a horizontal and vertical mirrored copy is made. Now there are
	4 images, including the original. Each of these four images is then
	mirrored along its two diagonal axes. So a total of 4 + 8 = 12 copies
	are returned. This also means an increase in required storage space by
	a factor 12.

	It is meant to be used with
	the chip-classification task, as the labels are not mirrored. This would
	result in wrongly placed labels for any other task than chip-classification.

	Args:
		aug_prob: (float between 0.0 and 1.0) probability that the data is mirrored.

	"""

	def __init__(self, aug_prob):
		self.aug_prob = aug_prob

	def process(self, training_data, tmp_dir):
		augmented_data = TrainingData() # Initiatlize empty training data object

		# Define functions that do the mirroring
		# Later these might include mirroring the labels as well
		def mirror_horizontally(chip,label=None):
			# Mirror label horizontally
			return np.flip(np.copy(chip), axis = 1)

		def mirror_vertically(chip,label=None):
			# Mirror label vertically
			return np.flip(np.copy(chip), axis = 0)

		def mirror_diagonal1(chip,label=None):
			# Mirror label diagonal 1
			# The mirror axis is from top left to bottom right
			return np.transpose(np.copy(chip), axes = [1,0,2])

		def mirror_diagonal2(chip,label=None):
			# Mirror label diagonal 2
			# The mirror axis is from bottom left to top right
			return np.transpose(np.copy(mirror_horizontally(chip)), axes = [1,0,2])

		for chip, window, labels in training_data:
			if random.uniform(0,1) < self.aug_prob:
				
				# Mirror along horizontal and vertical axes
				original = np.copy(chip)
				horizontal = mirror_horizontally(original)
				original_vertical = mirror_vertically(original)
				horizontal_vertical = mirror_vertically(horizontal)

				# For each of the above copies, make a mirror along
				# both diagonal axes.
				original_diag1 = mirror_diagonal1(original)
				original_diag2 = mirror_diagonal2(original)

				horizontal_diag1 = mirror_diagonal1(horizontal)
				horizontal_diag2 = mirror_diagonal2(horizontal)

				original_vertical_diag1 = mirror_diagonal1(original_vertical)
				original_vertical_diag2 = mirror_diagonal2(original_vertical)

				horizontal_vertical_diag1 = mirror_diagonal1(horizontal_vertical)
				horizontal_vertical_diag2 = mirror_diagonal2(horizontal_vertical)

				copies = [
					original,
					horizontal,
					original_vertical,
					horizontal_vertical,
					original_diag1,
					original_diag2,
					horizontal_diag1,
					horizontal_diag2,
					original_vertical_diag1,
					original_vertical_diag2,
					horizontal_vertical_diag1,
					horizontal_vertical_diag2
					]

				for copy in copies:
					# For each copy set the same window and labels as for the original
					# Note: the labels are not mirrored (yet)
					augmented_data.append(copy,window,labels)

		return augmented_data
