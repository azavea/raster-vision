import unittest
from os.path import join

import numpy as np

from rastervision.pipeline.file_system.utils import file_exists, get_tmp_dir
from rastervision.core.raster_stats import RasterStats
from rastervision.core.data import Scene
from rastervision.core.analyzer import StatsAnalyzerConfig
from tests.core.data.mock_raster_source import MockRasterSource

chip_sz = 300


def make_scene(i: int, is_random: bool = False
               ) -> tuple[Scene, MockRasterSource, np.ndarray]:
    rs = MockRasterSource([0, 1, 2], 3)
    img = np.zeros((600, 600, 3))
    img[:, :, 0] = 1 + i
    img[:, :, 1] = 2 + i
    img[:, :, 2] = 3 + i
    if not is_random:
        img[300:, 300:, :] = np.nan
    rs.set_raster(img)
    scene = Scene(str(i), rs)
    return scene, rs, img


class TestStatsAnalyzer(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = get_tmp_dir()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _test(self, is_random=False):
        sample_prob = 0.5

        scenes, raster_sources, imgs = zip(
            *[make_scene(i, is_random=is_random) for i in range(3)])

        imgs: np.ndarray = np.stack(imgs)
        pixels = imgs.reshape(-1, 3)
        exp_means = np.nanmean(pixels, axis=0)
        exp_stds = np.nanstd(pixels, axis=0)

        if is_random:
            analyzer_cfg = StatsAnalyzerConfig(
                output_uri=self.tmp_dir.name,
                chip_sz=chip_sz,
                sample_prob=sample_prob)
        else:
            analyzer_cfg = StatsAnalyzerConfig(
                output_uri=self.tmp_dir.name,
                chip_sz=chip_sz,
                sample_prob=None)
        analyzer = analyzer_cfg.build()
        analyzer.process(scenes, self.tmp_dir.name)

        stats = RasterStats.load(join(self.tmp_dir.name, 'stats.json'))
        np.testing.assert_array_almost_equal(stats.means, exp_means, decimal=3)
        np.testing.assert_array_almost_equal(stats.stds, exp_stds, decimal=3)
        if is_random:
            for rs in raster_sources:
                area = rs.extent.area
                exp_num_chips = round((area / (chip_sz**2)) * sample_prob)
                self.assertEqual(rs.mock._get_chip.call_count, exp_num_chips)

    def test_random(self):
        self._test(is_random=True)

    def test_sliding(self):
        self._test(is_random=False)

    def test_with_scene_group(self):
        scenes, _, _ = zip(*[make_scene(i) for i in range(3)])

        analyzer_cfg = StatsAnalyzerConfig(output_uri=self.tmp_dir.name)
        analyzer = analyzer_cfg.build(scene_group=('abc', set(range(3))))
        analyzer.process(scenes, self.tmp_dir.name)

        expected_stats_path = join(self.tmp_dir.name, 'abc', 'stats.json')
        self.assertTrue(file_exists(expected_stats_path, include_dir=False))


if __name__ == '__main__':
    unittest.main()
