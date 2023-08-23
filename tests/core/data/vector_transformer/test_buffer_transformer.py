import unittest

from shapely.geometry import (Polygon, Point, LineString, mapping, shape)

from rastervision.core.data.vector_transformer import (BufferTransformer,
                                                       BufferTransformerConfig)
from rastervision.core.data.utils import (geometry_to_feature,
                                          geometries_to_geojson)


class TestBufferTransformerConfig(unittest.TestCase):
    def test_build(self):
        cfg = BufferTransformerConfig(geom_type='Point')
        tf = cfg.build()
        self.assertIsInstance(tf, BufferTransformer)


class TestBufferTransformer(unittest.TestCase):
    def test_transform(self):
        class_bufs = {0: 5}
        properties = dict(class_id=0)

        # polygons
        tf = BufferTransformer(
            geom_type='Polygon', class_bufs=class_bufs, default_buf=1)
        geom_in = Polygon.from_bounds(0, 0, 10, 10)
        feat_in = geometry_to_feature(mapping(geom_in), properties)
        geojson_in = geometries_to_geojson([feat_in])
        geojson_out = tf(geojson_in)
        geom_out = shape(geojson_out['features'][0]['geometry'])
        self.assertTrue(geom_out.equals(geom_in.buffer(5)))

        # points
        tf = BufferTransformer(
            geom_type='Point', class_bufs=class_bufs, default_buf=1)
        geom_in = Point(0, 0)
        feat_in = geometry_to_feature(mapping(geom_in), properties)
        geojson_in = geometries_to_geojson([feat_in])
        geojson_out = tf(geojson_in)
        geom_out = shape(geojson_out['features'][0]['geometry'])
        self.assertTrue(geom_out.equals(geom_in.buffer(5)))

        # linestrings
        tf = BufferTransformer(
            geom_type='LineString', class_bufs=class_bufs, default_buf=1)
        geom_in = LineString([(0, 0), (1, 1), (2, 2)])
        feat_in = geometry_to_feature(mapping(geom_in), properties)
        geojson_in = geometries_to_geojson([feat_in])
        geojson_out = tf(geojson_in)
        geom_out = shape(geojson_out['features'][0]['geometry'])
        self.assertTrue(geom_out.equals(geom_in.buffer(5)))

        # mismatched geom_type
        tf = BufferTransformer(
            geom_type='Point', class_bufs=class_bufs, default_buf=1)
        geom_in = Polygon.from_bounds(0, 0, 10, 10)
        feat_in = geometry_to_feature(mapping(geom_in), properties)
        geojson_in = geometries_to_geojson([feat_in])
        geojson_out = tf(geojson_in)
        geom_out = shape(geojson_out['features'][0]['geometry'])
        self.assertTrue(geom_out.equals(geom_in))


if __name__ == '__main__':
    unittest.main()
