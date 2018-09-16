from copy import deepcopy

import rastervision as rv
from rastervision.augmentor.nodata_augmentor import NodataAugmentor
from rastervision.augmentor.augmentor_config \
    import (AugmentorConfig, AugmentorConfigBuilder)
from rastervision.protos.augmentor_pb2 import AugmentorConfig as AugmentorConfigMsg


class NodataAugmentorConfig(AugmentorConfig):
    def __init__(self, aug_prob=0.5):
        super().__init__(rv.NODATA_AUGMENTOR)
        self.aug_prob = aug_prob

    def to_proto(self):
        msg = AugmentorConfigMsg(
            augmentor_type=self.augmentor_type, aug_prob=self.aug_prob)
        return msg

    def create_augmentor(self):
        return NodataAugmentor(self.aug_prob)


class NodataAugmentorConfigBuilder(AugmentorConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {'aug_prob': prev.aug_prob}
        super().__init__(NodataAugmentorConfig, config)

    def from_proto(self, msg):
        return self.with_probablity(msg.aug_prob)

    def with_probability(self, aug_prob):
        """Sets the probability for this augmentation.

        Determines how probable this augmentation will happen
        to negative chips.

        Args:
           aug_prob: Float value between 0.0 and 1.0
        """
        b = deepcopy(self)
        b.config['aug_prob'] = aug_prob
        return b
