import unittest
import os

import numpy as np

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.raster_stats import RasterStats
from rastervision.core.data import StatsTransformerConfig


class MockRVPipelineConfig:
    analyze_uri = '/abc/def/analyze'


class TestStatsTransformerConfig(unittest.TestCase):
    def test_update(self):
        cfg = StatsTransformerConfig(stats_uri=None)
        cfg.update(MockRVPipelineConfig())
        self.assertEqual(cfg.stats_uri,
                         '/abc/def/analyze/stats/train_scenes/stats.json')

        cfg = StatsTransformerConfig(stats_uri=None, scene_group='group1')
        cfg.update(MockRVPipelineConfig())
        self.assertEqual(cfg.stats_uri,
                         '/abc/def/analyze/stats/group1/stats.json')

    def test_update_root(self):
        cfg = StatsTransformerConfig(stats_uri=None, scene_group='group1')
        cfg.update(MockRVPipelineConfig())
        self.assertEqual(cfg.stats_uri,
                         '/abc/def/analyze/stats/group1/stats.json')
        cfg.update_root('/path/to/bundle')
        self.assertEqual(cfg.stats_uri,
                         '/path/to/bundle/analyze/stats/group1/stats.json')


class TestStatsTransformer(unittest.TestCase):
    def test_stats_transformer(self):
        raster_stats = RasterStats()
        raster_stats.means = list(np.ones((4, )))
        raster_stats.stds = list(np.ones((4, )) * 2)

        with get_tmp_dir() as tmp_dir:
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
