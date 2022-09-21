import unittest

from shapely.geometry import (Point, mapping, shape)

from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from rastervision.core.data.vector_transformer import ShiftTransformer
from rastervision.core.data.utils import (geometry_to_feature,
                                          geometries_to_geojson)

from tests import data_file_path


class TestShiftTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.crs_transformer = RasterioCRSTransformer.from_uri(
            data_file_path('3857.tif'))

    def test_transform(self):
        """This only tests the directionality of the change."""
        geom_in = Point(0, 0)
        feat_in = geometry_to_feature(mapping(geom_in))
        geojson_in = geometries_to_geojson([feat_in])

        # x_shift = 1, y_shift = 0
        tf = ShiftTransformer(x_shift=1, y_shift=0)
        geojson_out = tf(geojson_in, crs_transformer=self.crs_transformer)
        geom_out = shape(geojson_out['features'][0]['geometry'])
        self.assertAlmostEqual(geom_out.y, geom_in.y)
        self.assertGreater(geom_out.x, geom_in.x)

        # x_shift = 0, y_shift = 1
        tf = ShiftTransformer(x_shift=0, y_shift=1)
        geojson_out = tf(geojson_in, crs_transformer=self.crs_transformer)
        geom_out = shape(geojson_out['features'][0]['geometry'])
        self.assertAlmostEqual(geom_out.x, geom_in.x)
        self.assertLess(geom_out.y, geom_in.y)


if __name__ == '__main__':
    unittest.main()
