import unittest

from rastervision.core.box import Box
from rastervision.core.data import (ClassConfig, RasterioSource, Scene)

from tests import data_file_path


class TestScene(unittest.TestCase):
    def setUp(self) -> None:
        self.class_config = ClassConfig(names=['class_1'])
        self.img_uri = data_file_path(
            'multi_raster_source/const_100_600x600.tiff')

    def test_raster_source_with_bbox(self):
        bbox = Box(100, 100, 200, 200)
        rs = RasterioSource(self.img_uri, bbox=bbox)
        scene = Scene(id='', raster_source=rs)
        self.assertEqual(scene.bbox, bbox)
        self.assertEqual(scene.extent, bbox.extent)

    def test_aoi_polygons(self):
        bbox = Box(100, 100, 200, 200)
        rs = RasterioSource(self.img_uri, bbox=bbox)

        # w/o AOI
        scene = Scene(id='', raster_source=rs)
        self.assertListEqual(scene.aoi_polygons, [])
        self.assertListEqual(scene.aoi_polygons_bbox_coords, [])

        # w/ AOI
        aoi_polygons = [
            Box(50, 50, 150, 150).to_shapely(),
            Box(150, 150, 250, 250).to_shapely(),
            Box(300, 300, 400, 400).to_shapely(),
        ]
        aoi_polygons_bbox_coords = [
            Box(-50, -50, 50, 50).to_shapely(),
            Box(50, 50, 150, 150).to_shapely(),
        ]
        scene = Scene(id='', raster_source=rs, aoi_polygons=aoi_polygons)
        self.assertListEqual(scene.aoi_polygons, aoi_polygons[:2])
        self.assertListEqual(scene.aoi_polygons_bbox_coords,
                             aoi_polygons_bbox_coords)


if __name__ == '__main__':
    unittest.main()
