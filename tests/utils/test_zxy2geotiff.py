import unittest
import tempfile
from os.path import join

import mercantile
from PIL import Image
import numpy as np
import rasterio
from rastervision.utils.files import make_dir
from rastervision.utils.zxy2geotiff import _zxy2geotiff, merc2lnglat


def gen_zxy_tiles(root_dir, zoom, xlen, ylen, use_tms=False):
    img_arr = np.random.randint(
        0, 256, (ylen * 256, xlen * 256, 3), dtype=np.uint8)

    i = 0
    for y in range(ylen):
        for x in range(xlen):
            im = Image.fromarray(
                img_arr[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256, :])

            tile_y = y
            # The TMS convention is for the y axis to start at the bottom
            # rather than the top.
            if use_tms:
                tile_y = (2**zoom) - y - 1
            im_path = join(root_dir, '{}/{}/{}.png'.format(zoom, x, tile_y))
            make_dir(im_path, use_dirname=True)
            im.save(im_path)
            i += 1

    return img_arr


class TestZXY2Geotiff(unittest.TestCase):
    def setUp(self):
        tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = tmp_dir_obj.name
        self.tmp_dir = '/opt/data/test-zxy'

    def _test_zxy2geotiff(self, use_tms=False, make_cog=False):
        # We generate a 3x3 grid of zxy tiles and save them. Then,
        # get the lng/lat of the center of the NW (northwest) and SE tiles,
        # and pass those as bounds to zxy2geotiff. We open the resulting
        # geotiff and check that the content is correct.

        zoom = 18
        img_arr = gen_zxy_tiles(self.tmp_dir, zoom, 3, 3, use_tms=use_tms)
        # Get center of NW and SE tiles.
        nw_bounds = mercantile.xy_bounds(0, 0, zoom)
        nw_merc_y = nw_bounds.bottom + (nw_bounds.top - nw_bounds.bottom) / 2
        nw_merc_x = nw_bounds.left + (nw_bounds.right - nw_bounds.left) / 2
        nw_lng, nw_lat = merc2lnglat(nw_merc_x, nw_merc_y)

        se_bounds = mercantile.xy_bounds(2, 2, zoom)
        se_merc_y = se_bounds.bottom + (se_bounds.top - se_bounds.bottom) / 2
        se_merc_x = se_bounds.left + (se_bounds.right - se_bounds.left) / 2
        se_lng, se_lat = merc2lnglat(se_merc_x, se_merc_y)

        # min_lat, min_lng, max_lat, max_lng = bounds
        bounds = [se_lat, nw_lng, nw_lat, se_lng]
        output_uri = join(self.tmp_dir, 'output.tif')
        tile_schema = join(self.tmp_dir, '{z}/{x}/{y}.png')
        if use_tms:
            tile_schema = join(self.tmp_dir, '{z}/{x}/{-y}.png')
        _zxy2geotiff(tile_schema, zoom, bounds, output_uri, make_cog=make_cog)

        with rasterio.open(output_uri) as dataset:
            tiff_arr = dataset.read()
            self.assertEqual(tiff_arr.shape, (3, 512, 512))
            exp_arr = np.transpose(img_arr, (2, 0, 1))[:, 128:-128, 128:-128]
            np.testing.assert_array_equal(tiff_arr, exp_arr)

    def test_zxy2geotiff(self):
        self._test_zxy2geotiff()

    def test_zxy2geotiff_cog(self):
        self._test_zxy2geotiff(make_cog=True)

    def test_zxy2geotiff_tms(self):
        self._test_zxy2geotiff(use_tms=True)

    def test_zxy2geotiff_dry_run(self):
        zoom = 18

        # Get center of NW and SE tiles.
        nw_bounds = mercantile.xy_bounds(0, 0, zoom)
        nw_merc_y = nw_bounds.bottom + (nw_bounds.top - nw_bounds.bottom) / 2
        nw_merc_x = nw_bounds.left + (nw_bounds.right - nw_bounds.left) / 2
        nw_lng, nw_lat = merc2lnglat(nw_merc_x, nw_merc_y)

        se_bounds = mercantile.xy_bounds(2, 2, zoom)
        se_merc_y = se_bounds.bottom + (se_bounds.top - se_bounds.bottom) / 2
        se_merc_x = se_bounds.left + (se_bounds.right - se_bounds.left) / 2
        se_lng, se_lat = merc2lnglat(se_merc_x, se_merc_y)

        # min_lat, min_lng, max_lat, max_lng = bounds
        bounds = [se_lat, nw_lng, nw_lat, se_lng]
        output_uri = join(self.tmp_dir, 'output.tif')
        tile_schema = join(self.tmp_dir, '{z}/{x}/{y}.png')
        height, width, transform = _zxy2geotiff(
            tile_schema, zoom, bounds, output_uri, dry_run=True)
        self.assertEqual((height, width), (512, 512))

    def test_zxy2geotiff_single_tile(self):
        # Same as above test except it uses a single tile instead of a 3x3
        # grid.
        zoom = 18
        img_arr = gen_zxy_tiles(self.tmp_dir, zoom, 1, 1)
        tile_schema = join(self.tmp_dir, '{z}/{x}/{y}.png')

        # Get NW and SE corner of central half of tile.
        nw_bounds = mercantile.xy_bounds(0, 0, zoom)
        nw_merc_y = nw_bounds.top - (nw_bounds.top - nw_bounds.bottom) / 4
        nw_merc_x = nw_bounds.left + (nw_bounds.right - nw_bounds.left) / 4
        nw_lng, nw_lat = merc2lnglat(nw_merc_x, nw_merc_y)

        se_bounds = mercantile.xy_bounds(0, 0, zoom)
        se_merc_y = se_bounds.bottom + (se_bounds.top - se_bounds.bottom) / 4
        se_merc_x = se_bounds.right - (se_bounds.right - se_bounds.left) / 4
        se_lng, se_lat = merc2lnglat(se_merc_x, se_merc_y)

        # min_lat, min_lng, max_lat, max_lng = bounds
        bounds = [se_lat, nw_lng, nw_lat, se_lng]
        output_uri = join(self.tmp_dir, 'output.tif')
        _zxy2geotiff(tile_schema, zoom, bounds, output_uri)

        with rasterio.open(output_uri) as dataset:
            tiff_arr = dataset.read()
            self.assertEqual(tiff_arr.shape, (3, 128, 128))
            exp_arr = np.transpose(img_arr, (2, 0, 1))[:, 64:-64, 64:-64]
            np.testing.assert_array_equal(tiff_arr, exp_arr)


if __name__ == '__main__':
    unittest.main()
