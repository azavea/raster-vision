import unittest
import tempfile
from os.path import join

import numpy as np
import mercantile

from rastervision.utils.zxy2geotiff import merc2lnglat
from rastervision.data import RasterSourceConfig
import rastervision as rv
from tests.utils.test_zxy2geotiff import gen_zxy_tiles


class TestZXYRasterSource(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = self.tmp_dir_obj.name

    def build_source(self, tile_schema, zoom, bounds):
        config = RasterSourceConfig.builder(rv.ZXY_RASTER_SOURCE) \
            .with_tile_schema(tile_schema) \
            .with_zoom(zoom) \
            .with_bounds(bounds) \
            .build()

        config = RasterSourceConfig.builder(rv.ZXY_RASTER_SOURCE) \
            .from_proto(config.to_proto()) \
            .build()

        source = config.create_source(self.tmp_dir)
        return source

    def test_zxy_raster_source(self):
        zoom = 18
        xlen = 4
        ylen = 4
        img_arr = gen_zxy_tiles(self.tmp_dir, zoom, xlen, ylen)

        nw_bounds = mercantile.xy_bounds(0, 0, zoom)
        nw_lng, nw_lat = merc2lnglat(nw_bounds.left, nw_bounds.top)
        se_bounds = mercantile.xy_bounds(2, 2, zoom)
        se_lng, se_lat = merc2lnglat(se_bounds.right, se_bounds.bottom)
        bounds = [se_lat, nw_lng, nw_lat, se_lng]

        tile_schema = join(self.tmp_dir, '{z}/{x}/{y}.png')
        raster_source = self.build_source(tile_schema, zoom, bounds)

        with raster_source.activate():
            out_img_arr = raster_source.get_image_array()
            np.testing.assert_array_equal(out_img_arr,
                                          img_arr[0:768, 0:768, :])


if __name__ == '__main__':
    unittest.main()
