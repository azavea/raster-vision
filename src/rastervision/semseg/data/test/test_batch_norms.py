import numpy as np
import unittest

from rastervision.semseg.data.potsdam import (PotsdamNumpyFileGenerator,
                                              PotsdamImageFileGenerator)
from rastervision.common.settings import datasets_path


class NormalizationTestCase(unittest.TestCase):
    def test_numpy_potsdam_batch(self):
        generator = PotsdamNumpyFileGenerator(datasets_path, [0, 1, 2, 3, 4])
        means, stds = generator.compute_channel_stats(100, True)

        # passes when mean = 0 with an error of +/- 0.3 and stds = 1 +/- 0.3.
        self.assertTrue((np.absolute(means) < 0.3).all())
        self.assertTrue((stds > 0.3).all() and (stds < 1.3).all())

    def test_image_potsdam_batch(self):
        generator = PotsdamImageFileGenerator(datasets_path, [0, 1, 2, 3, 4])
        means, stds = generator.compute_channel_stats(100, True)

        # passes when mean = 0 and stds = 1 with an error of +/- 0.3.
        self.assertTrue((np.absolute(means) < 0.3).all())
        self.assertTrue((stds > 0.7).all() and (stds < 1.3).all())


if __name__ == '__main__':
    unittest.main()
