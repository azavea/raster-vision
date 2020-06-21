import unittest
import os

import numpy as np

from rastervision.core.data import RasterStats, StatsTransformerConfig
from rastervision.pipeline import rv_config


class TestRasterTransformer(unittest.TestCase):
    def test_stats_transformer(self):
        raster_stats = RasterStats()
        raster_stats.means = list(np.ones((4, )))
        raster_stats.stds = list(np.ones((4, )) * 2)

        with rv_config.get_tmp_dir() as tmp_dir:
            stats_uri = os.path.join(tmp_dir, 'stats.json')
            raster_stats.save(stats_uri)

            # All values have z-score of 1, which translates to
            # uint8 value of 170.
            transformer = StatsTransformerConfig(stats_uri=stats_uri).build()
            chip = np.ones((2, 2, 4)) * 3
            out_chip = transformer.transform(chip)
            expected_out_chip = np.ones((2, 2, 4)) * 170
            np.testing.assert_equal(out_chip, expected_out_chip)


if __name__ == '__main__':
    unittest.main()
