import unittest
import os
import json
import copy

import shapely

import rastervision as rv
from rastervision.rv_config import RVConfig
from rastervision.data.label_source import (infer_cell, infer_labels,
                                            read_labels)
from rastervision.data.crs_transformer import IdentityCRSTransformer
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.data.utils import geojson_to_shapes
from tests import data_file_path
from tests.data.mock_crs_transformer import DoubleCRSTransformer


class TestChipClassificationLabelSource(unittest.TestCase):
    def setUp(self):
        self.crs_transformer = DoubleCRSTransformer()
        # Use a multipolygon with two polygons that are the same to test that
        # multipolygons can be handled.
        self.geojson = {
            'type':
            'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'geometry': {
                    'type':
                    'MultiPolygon',
                    'coordinates':
                    [[[[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0., 0.]]],
                     [[[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0., 0.]]]]
                },
                'properties': {
                    'class_name': 'car',
                    'class_id': 1,
                    'score': 0.0
                }
            }, {
                'type': 'Feature',
                'geometry': {
                    'type':
                    'Polygon',
                    'coordinates': [[[1., 1.], [1., 2.], [2., 2.], [2., 1.],
                                     [1., 1.]]]
                },
                'properties': {
                    'score': 0.0,
                    'class_name': 'house',
                    'class_id': 2
                }
            }]
        }

        # Make copy of geojson with multipolygon converted to polygon. This will be used
        # to test read_labels.
        self.geojson_no_multipolygons = copy.deepcopy(self.geojson)
        feature = self.geojson_no_multipolygons['features'][0]
        feature['geometry']['type'] = 'Polygon'
        feature['geometry']['coordinates'] = feature['geometry'][
            'coordinates'][0]

        self.class_map = ClassMap([ClassItem(1, 'car'), ClassItem(2, 'house')])

        self.box1 = Box.make_square(0, 0, 2)
        self.box2 = Box.make_square(2, 2, 2)
        self.class_id1 = 1
        self.class_id2 = 2
        self.background_class_id = 3

        self.shapes = geojson_to_shapes(self.geojson, self.crs_transformer)

        self.file_name = 'labels.json'
        self.temp_dir = RVConfig.get_tmp_dir()
        self.file_path = os.path.join(self.temp_dir.name, self.file_name)

        with open(self.file_path, 'w') as label_file:
            self.geojson_str = json.dumps(self.geojson)
            label_file.write(self.geojson_str)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_str_tree(self):
        str_tree = shapely.strtree.STRtree(
            [shape for shape, class_id in self.shapes])
        # Monkey-patching class_id onto shapely.geom is not a good idea because
        # if you transform it, the class_id will be lost, but this works here. I wanted to
        # use a dictionary to associate shape with class_id, but couldn't because they are
        # mutable.
        for shape, class_id in self.shapes:
            shape.class_id = class_id

        # Check first box.
        query_box = Box.make_square(0, 0, 1)
        query_geom = shapely.geometry.Polygon(
            [(p[0], p[1]) for p in query_box.geojson_coordinates()])
        polygons = str_tree.query(query_geom)

        self.assertEqual(len(polygons), 2)
        self.assertEqual(Box.from_shapely(polygons[0]), self.box1)
        self.assertEqual(polygons[0].class_id, self.class_id1)

        # Check second box.
        query_box = Box.make_square(3, 3, 1)
        query_geom = shapely.geometry.Polygon(
            [(p[0], p[1]) for p in query_box.geojson_coordinates()])
        polygons = str_tree.query(query_geom)

        self.assertEqual(len(polygons), 1)
        self.assertEqual(Box.from_shapely(polygons[0]), self.box2)
        self.assertEqual(polygons[0].class_id, self.class_id2)

    def test_infer_cell1(self):
        # More of box 1 is in cell.
        cell = Box.make_square(0, 0, 3)
        ioa_thresh = 0.5
        use_intersection_over_cell = False
        background_class_id = None
        pick_min_class_id = False

        class_id = infer_cell(self.shapes, cell, ioa_thresh,
                              use_intersection_over_cell, background_class_id,
                              pick_min_class_id)
        self.assertEqual(class_id, self.class_id1)

    def test_infer_cell2(self):
        # More of box 2 is in cell.
        cell = Box.make_square(1, 1, 3)
        ioa_thresh = 0.5
        use_intersection_over_cell = False
        background_class_id = None
        pick_min_class_id = False

        class_id = infer_cell(self.shapes, cell, ioa_thresh,
                              use_intersection_over_cell, background_class_id,
                              pick_min_class_id)
        self.assertEqual(class_id, self.class_id2)

    def test_infer_cell3(self):
        # Only box 2 is in cell, but IOA isn't high enough.
        cell = Box.make_square(3, 3, 3)
        ioa_thresh = 0.5
        use_intersection_over_cell = False
        background_class_id = None
        pick_min_class_id = False

        class_id = infer_cell(self.shapes, cell, ioa_thresh,
                              use_intersection_over_cell, background_class_id,
                              pick_min_class_id)
        self.assertEqual(class_id, None)

    def test_infer_cell4(self):
        # Both boxes inside cell, but using intersection_over_cell,
        # the IOA isn't high enough.
        cell = Box.make_square(0, 0, 10)
        ioa_thresh = 0.5
        use_intersection_over_cell = True
        background_class_id = None
        pick_min_class_id = False

        class_id = infer_cell(self.shapes, cell, ioa_thresh,
                              use_intersection_over_cell, background_class_id,
                              pick_min_class_id)
        self.assertEqual(class_id, None)

    def test_infer_cell5(self):
        # More of box1 in cell, using intersection_over_cell with the
        # IOA high enough.
        cell = Box.make_square(0, 0, 3)
        ioa_thresh = 0.4
        use_intersection_over_cell = True
        background_class_id = None
        pick_min_class_id = False

        class_id = infer_cell(self.shapes, cell, ioa_thresh,
                              use_intersection_over_cell, background_class_id,
                              pick_min_class_id)
        self.assertEqual(class_id, self.class_id1)

    def test_infer_cell6(self):
        # No boxes overlap enough, use background_class_id
        cell = Box.make_square(0, 0, 10)
        ioa_thresh = 0.5
        use_intersection_over_cell = True
        background_class_id = self.background_class_id
        pick_min_class_id = False

        class_id = infer_cell(self.shapes, cell, ioa_thresh,
                              use_intersection_over_cell, background_class_id,
                              pick_min_class_id)
        self.assertEqual(class_id, self.background_class_id)

    def test_infer_cell7(self):
        # Cell doesn't overlap with any boxes.
        cell = Box.make_square(10, 10, 1)
        ioa_thresh = 0.5
        use_intersection_over_cell = True
        background_class_id = None
        pick_min_class_id = False

        class_id = infer_cell(self.shapes, cell, ioa_thresh,
                              use_intersection_over_cell, background_class_id,
                              pick_min_class_id)
        self.assertEqual(class_id, None)

    def test_infer_cell8(self):
        # box2 overlaps more than box1, but using pick_min_class_id, so
        # picks box1.
        cell = Box.make_square(1, 1, 3)
        ioa_thresh = 0.5
        use_intersection_over_cell = False
        background_class_id = None
        pick_min_class_id = True

        class_id = infer_cell(self.shapes, cell, ioa_thresh,
                              use_intersection_over_cell, background_class_id,
                              pick_min_class_id)
        self.assertEqual(class_id, self.class_id2)

    def test_infer_labels(self):
        extent = Box.make_square(0, 0, 4)
        ioa_thresh = 0.5
        use_intersection_over_cell = False
        background_class_id = self.background_class_id
        pick_min_class_id = False
        cell_size = 2

        labels = infer_labels(
            self.geojson, self.crs_transformer, extent, cell_size, ioa_thresh,
            use_intersection_over_cell, pick_min_class_id, background_class_id)
        cells = labels.get_cells()

        self.assertEqual(len(cells), 4)
        class_id = labels.get_cell_class_id(self.box1)
        self.assertEqual(class_id, self.class_id1)
        class_id = labels.get_cell_class_id(self.box2)
        self.assertEqual(class_id, self.class_id2)
        class_id = labels.get_cell_class_id(Box.make_square(0, 2, 2))
        self.assertEqual(class_id, self.background_class_id)
        class_id = labels.get_cell_class_id(Box.make_square(2, 0, 2))
        self.assertEqual(class_id, self.background_class_id)

    def test_read_labels1(self):
        # Extent only has enough of first box in it.
        extent = Box.make_square(0, 0, 0.5)
        labels = read_labels(self.geojson_no_multipolygons,
                             self.crs_transformer, extent)

        cells = labels.get_cells()
        self.assertEqual(len(cells), 1)
        class_id = labels.get_cell_class_id(self.box1)
        self.assertEqual(class_id, self.class_id1)
        class_id = labels.get_cell_class_id(self.box2)
        self.assertEqual(class_id, None)

    def test_read_labels2(self):
        # Extent contains both boxes.
        extent = Box.make_square(0, 0, 4)
        labels = read_labels(self.geojson_no_multipolygons,
                             self.crs_transformer, extent)

        cells = labels.get_cells()
        self.assertEqual(len(cells), 2)
        class_id = labels.get_cell_class_id(self.box1)
        self.assertEqual(class_id, self.class_id1)
        class_id = labels.get_cell_class_id(self.box2)
        self.assertEqual(class_id, self.class_id2)

    def test_missing_config_uri(self):
        with self.assertRaises(rv.ConfigError):
            rv.data.ChipClassificationLabelSourceConfig.builder(
                rv.CHIP_CLASSIFICATION).build()

    def test_no_missing_config(self):
        try:
            rv.data.ChipClassificationLabelSourceConfig.builder(
                rv.CHIP_CLASSIFICATION).with_uri('x.geojson').build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')

    def test_deprecated_builder(self):
        try:
            rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
              .with_uri('x.geojson') \
              .build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')

    def test_builder(self):
        uri = data_file_path('polygon-labels.geojson')
        msg = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION) \
                .with_uri(uri) \
                .build().to_proto()
        config = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION) \
                   .from_proto(msg).build()
        self.assertEqual(config.vector_source.uri, uri)

        classes = ['one', 'two']
        extent = Box.make_square(0, 0, 10)
        crs_transformer = IdentityCRSTransformer()
        with RVConfig.get_tmp_dir() as tmp_dir:
            task_config = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                            .with_classes(classes) \
                            .build()
            config.create_source(task_config, extent, crs_transformer, tmp_dir)


if __name__ == '__main__':
    unittest.main()
