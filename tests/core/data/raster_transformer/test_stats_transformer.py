import unittest
from os.path import join

import numpy as np

from rastervision.pipeline.config import build_config
from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.raster_stats import RasterStats
from rastervision.core.data import (RasterioSource, StatsTransformer,
                                    StatsTransformerConfig)
from rastervision.core.data.raster_transformer.stats_transformer_config import (  # noqa
    stats_transformer_config_upgrader)

from tests import data_file_path


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

    def test_upgrader_v2(self):
        cfg = StatsTransformerConfig()
        old_cfg_dict = cfg.dict()
        old_cfg_dict.pop('needs_channel_order')
        new_cfg_dict = stats_transformer_config_upgrader(old_cfg_dict, 2)
        self.assertEqual(new_cfg_dict['scene_group'], '__N/A__')

    def test_upgrader_v13(self):
        stats = RasterStats(np.array([1, 2]), np.array([3, 4]))

        with get_tmp_dir() as tmp_dir:
            stats_uri = join(tmp_dir, 'stats.json')
            stats.save(stats_uri)

            cfg = StatsTransformerConfig(stats_uri=stats_uri)
            old_cfg_dict = cfg.dict()
            old_cfg_dict.pop('needs_channel_order')
            new_cfg_dict = stats_transformer_config_upgrader(old_cfg_dict, 13)
            self.assertTrue(new_cfg_dict['needs_channel_order'])

            cfg = build_config(new_cfg_dict)
            self.assertIsInstance(cfg, StatsTransformerConfig)

            tf = cfg.build(channel_order=[1, 0])
            np.testing.assert_array_equal(tf.means, np.array([2, 1]))
            np.testing.assert_array_equal(tf.stds, np.array([4, 3]))


class TestStatsTransformer(unittest.TestCase):
    def test_transform(self):
        # All values have z-score of 1, which translates to
        # uint8 value of 170.
        tf = StatsTransformer(np.ones((4, )), np.ones((4, )) * 2)
        chip_in = np.ones((2, 2, 4)) * 3
        chip_out = tf.transform(chip_in)
        chip_out_expected = np.ones((2, 2, 4)) * 170
        np.testing.assert_equal(chip_out, chip_out_expected)

    def test_transform_noop(self):
        tf = StatsTransformer(np.ones((4, )), np.ones((4, )) * 2)
        chip_in = np.ones((2, 2, 4), dtype=np.uint8)
        chip_out = tf.transform(chip_in)
        np.testing.assert_equal(chip_out, chip_in)

    def test_stats(self):
        tf = StatsTransformer([1, 2], [3, 4])
        stats = tf.stats
        self.assertIsInstance(stats, RasterStats)
        np.testing.assert_array_equal(stats.means, np.array([1, 2]))
        np.testing.assert_array_equal(stats.stds, np.array([3, 4]))

    def test_from_raster_stats(self):
        stats = RasterStats(np.array([1, 2]), np.array([3, 4]))
        tf = StatsTransformer.from_raster_stats(stats, channel_order=[1, 0])
        np.testing.assert_array_equal(tf.means, np.array([2, 1]))
        np.testing.assert_array_equal(tf.stds, np.array([4, 3]))

    def test_from_stats_json(self):
        stats = RasterStats(np.array([1, 2]), np.array([3, 4]))
        with get_tmp_dir() as tmp_dir:
            stats_uri = join(tmp_dir, 'stats.json')
            stats.save(stats_uri)
            tf = StatsTransformer.from_stats_json(stats_uri)
            np.testing.assert_array_equal(tf.means, np.array([1, 2]))
            np.testing.assert_array_equal(tf.stds, np.array([3, 4]))

    def test_from_raster_sources(self):
        rs = RasterioSource(data_file_path('ones.tif'))
        tf = StatsTransformer.from_raster_sources([rs])
        self.assertTrue(np.all(tf.means == 1))
        self.assertTrue(np.all(tf.stds == 0))

    def test_get_out_dtype(self):
        tf = StatsTransformer(np.zeros((4, )), np.ones((4, )))
        self.assertEqual(tf.get_out_dtype(np.float32), np.uint8)
        self.assertEqual(tf.get_out_dtype(np.uint8), np.uint8)

    def test_get_out_channels(self):
        tf = StatsTransformer(np.zeros((4, )), np.ones((4, )))
        self.assertEqual(tf.get_out_channels(3), 3)
        self.assertEqual(tf.get_out_channels(8), 8)


if __name__ == '__main__':
    unittest.main()
