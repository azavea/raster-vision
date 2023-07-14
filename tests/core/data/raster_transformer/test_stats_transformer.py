import unittest
from os.path import join

import numpy as np

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.raster_stats import RasterStats
from rastervision.core.data import StatsTransformer, StatsTransformerConfig


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

    def test_build(self):
        stats = RasterStats(np.array([1, 2]), np.array([3, 4]))

        with get_tmp_dir() as tmp_dir:
            stats_uri = join(tmp_dir, 'stats.json')
            stats.save(stats_uri)
            tf = StatsTransformerConfig(stats_uri=stats_uri).build()
            np.testing.assert_array_equal(tf.means, np.array([1, 2]))
            np.testing.assert_array_equal(tf.stds, np.array([3, 4]))


class TestStatsTransformer(unittest.TestCase):
    def test_transform(self):
        # All values have z-score of 1, which translates to
        # uint8 value of 170.
        tf = StatsTransformer(np.ones((4, )), np.ones((4, )) * 2)
        chip = np.ones((2, 2, 4)) * 3
        out_chip = tf.transform(chip)
        expected_out_chip = np.ones((2, 2, 4)) * 170
        np.testing.assert_equal(out_chip, expected_out_chip)

    def test_stats(self):
        tf = StatsTransformer([1, 2], [3, 4])
        stats = tf.stats
        self.assertIsInstance(stats, RasterStats)
        np.testing.assert_array_equal(stats.means, np.array([1, 2]))
        np.testing.assert_array_equal(stats.stds, np.array([3, 4]))

    def test_from_raster_stats(self):
        stats = RasterStats(np.array([1, 2]), np.array([3, 4]))
        tf = StatsTransformer.from_raster_stats(stats)
        np.testing.assert_array_equal(tf.means, np.array([1, 2]))
        np.testing.assert_array_equal(tf.stds, np.array([3, 4]))

    def test_from_stats_json(self):
        stats = RasterStats(np.array([1, 2]), np.array([3, 4]))
        with get_tmp_dir() as tmp_dir:
            stats_uri = join(tmp_dir, 'stats.json')
            stats.save(stats_uri)
            tf = StatsTransformer.from_stats_json(stats_uri)
            np.testing.assert_array_equal(tf.means, np.array([1, 2]))
            np.testing.assert_array_equal(tf.stds, np.array([3, 4]))


if __name__ == '__main__':
    unittest.main()
