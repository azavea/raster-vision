from copy import deepcopy

import rastervision as rv
from rastervision.augmentor.mirror_augmentor import MirrorAugmentor
from rastervision.augmentor.augmentor_config \
	import (AugmentorConfig, AugmentorConfigBuilder)
from rastervision.protos.augmentor_pb2 import AugmentorConfig as AugmentorConfigMsg


class MirrorAugmentorConfig(AugmentorConfig):
	def __init__(self, aug_prob=1.0):
		super().__init__(rv.MIRROR_AUGMENTOR)
		self.aug_prob = aug_prob

	def to_proto(self):
		msg = AugmentorConfigMsg(
			augmentor_type = 
				self.augmentor_type,
				aug_prob = self.aug_prob,
		)
		return msg

	def create_augmentor(self):
		return MirrorAugmentor(
			self.aug_prob,
			self.axes
		)

	def report_io(self, command_type, io_def):
		pass

class MirrorAugmentorConfigBuilder(AugmentorConfigBuilder):
	def __init__(self, prev=None):
		config = {}
		if prev:
			config = {
				'aug_prob': prev.aug_prob
			}
		super().__init__(MirrorAugmentorConfig, config)

	def from_proto(self, msg):
		a = self.with_probability(msg.aug_prob)
		return a

	def with_probability(self, aug_prob):
		'''Sets the probability for this augmentation.

		Determines how probable it is that this augmentation will happen to all chips.
		Since this augmentation is usually applied when ther is little training data 
		available, the default is 1.0.
		'''
		b = deepcopy(self)
		b.config['aug_prob'] = aug_prob
		return b

	def with_axes(self, axes):
		'''Sets the axis along which the mirrorring that is done.

		Default is 4, meaning that mirroring is done
		along the x and y axis, which is a horizontal and vertical mirror.
		If set to 8 also both the diagonal axis are used. Beware that this
		also increases the amount of data by a factor 8, so ensure
		you have enough storage available.
		'''
		b = deepcopy(self)
		b.config['axes'] = axes
		return b