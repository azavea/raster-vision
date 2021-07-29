import unittest
from itertools import product

import numpy as np
from scipy.stats import chisquare
from shapely.geometry import Polygon, MultiPolygon, MultiPoint

from rastervision.pytorch_learner.dataset.utils import AoiSampler


class TestAoiSampler(unittest.TestCase):
    def test_sampler(self, nsamples: int = 200):
        """This test attempts to check if the points are distributed uniformly
        within the AOI.

        To do this, it performs the following steps:
        - Create an AOI in the form of a plus-sign shape centered in a 6x6
        grid.
        - Create an AoiSampler for this AOI.
        - Break the AOI up into small 1x1 blocks.
        - Use the AoiSampler to sample nsamples points.
        - Count the points that fall in each block.
        - Perform a Chi-squared test to see how the counts compare to the
        expected counts (= nsamples / nblocks).

        The null hypothesis for the Chi-squared test is that the counts are
        distributed identically to the expected counts. A low enough p-value
        (lower than the significance level) would indicate that the null
        hypothesis must be rejected, meaning the points are not distributed
        uniformly inside the AOI. This will cause this unit test to fail. We
        use a significance level of 0.05 here.

        Args:
            nsamples (int, optional): Number of points to sample. It is
                important for the sample size to not be too large or the test
                will become over-powered.
                Defaults to 200.
        """
        np.random.seed(0)

        # create a polygon shaped like a plus-sign
        hbar = Polygon.from_bounds(xmin=0, ymin=2, xmax=6, ymax=4)
        vbar = Polygon.from_bounds(xmin=2, ymin=0, xmax=4, ymax=6)
        polygons = MultiPolygon([hbar, vbar])

        aoi_sampler = AoiSampler(polygons)

        # break the polygon up into small blocks of 1x1
        blocks = []
        sz = 1
        for x, y in product(np.arange(0, 6, sz), np.arange(0, 6, sz)):
            b = Polygon.from_bounds(x, y, x + sz, y + sz)
            if b.within(hbar) or b.within(vbar):
                blocks.append(b)
        blocks = MultiPolygon(blocks)

        points = MultiPoint(aoi_sampler.sample(n=nsamples))
        # number of points in each block
        counts = np.array(
            [len(block.intersection(points)) for block in blocks])

        p_value = chisquare(counts).pvalue

        self.assertEqual(counts.sum(), nsamples)
        self.assertGreaterEqual(p_value, 0.05)


if __name__ == '__main__':
    unittest.main()
