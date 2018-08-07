import unittest
import tempfile
import os
import json

from shapely import geometry

from rastervision.label_stores.classification_geojson_file import (
    ClassificationGeoJSONFile, get_str_tree, infer_cell, infer_labels,
    read_labels, to_geojson)
from rastervision.core.crs_transformer import CRSTransformer
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.protos.label_store_pb2 import (
    ClassificationGeoJSONFile as ClassificationGeoJSONFileConfig)


class DoubleCRSTransformer(CRSTransformer):
    """Mock CRSTransformer used for testing.

    Assumes map coords are 2x pixels coords.
    """

    def map_to_pixel(self, web_point):
        return (web_point[0] * 2, web_point[1] * 2)

    def pixel_to_map(self, pixel_point):
        return (pixel_point[0] / 2, pixel_point[1] / 2)


class TestObjectDetectionJsonFile(unittest.TestCase):
    def setUp(self):
        self.crs_transformer = DoubleCRSTransformer()
        self.geojson_dict = {
            'type':
            'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'geometry': {
                    'type':
                    'Polygon',
                    'coordinates': [[[0., 0.], [0., 1.], [1., 1.], [1., 0.],
                                     [0., 0.]]]
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

        self.aoi_dict = {
            'type':
            'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'geometry': {
                    'type':
                    'Polygon',
                    'coordinates': [[[1., 0.], [1., 1.], [0., 1.], [0., 0.],
                                     [1., 0.]]]
                },
                'properties': {}
            }]
        }

        self.class_map = ClassMap([ClassItem(1, 'car'), ClassItem(2, 'house')])

        self.box1 = Box.make_square(0, 0, 2)
        self.box2 = Box.make_square(2, 2, 2)
        self.class_id1 = 1
        self.class_id2 = 2
        self.background_class_id = 3

        self.str_tree = get_str_tree(self.geojson_dict, self.crs_transformer)

        self.file_name = 'labels.json'
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, self.file_name)

        with open(self.file_path, 'w') as label_file:
            self.geojson_str = json.dumps(self.geojson_dict)
            label_file.write(self.geojson_str)

        self.aoi_file_name = 'aoi.json'
        self.aoi_file_path = os.path.join(self.temp_dir.name,
                                          self.aoi_file_name)

        with open(self.aoi_file_path, 'w') as aoi_file:
            self.aoi_str = json.dumps(self.aoi_dict)
            aoi_file.write(self.aoi_str)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_str_tree(self):
        # Check first box.
        query_box = Box.make_square(0, 0, 1)
        query_geom = geometry.Polygon(
            [(p[0], p[1]) for p in query_box.geojson_coordinates()])
        polygons = self.str_tree.query(query_geom)

        self.assertEqual(len(polygons), 1)
        self.assertEqual(Box.from_shapely(polygons[0]), self.box1)
        self.assertEqual(polygons[0].class_id, self.class_id1)

        # Check second box.
        query_box = Box.make_square(3, 3, 1)
        query_geom = geometry.Polygon(
            [(p[0], p[1]) for p in query_box.geojson_coordinates()])
        polygons = self.str_tree.query(query_geom)

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

        class_id = infer_cell(self.str_tree, cell, ioa_thresh,
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

        class_id = infer_cell(self.str_tree, cell, ioa_thresh,
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

        class_id = infer_cell(self.str_tree, cell, ioa_thresh,
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

        class_id = infer_cell(self.str_tree, cell, ioa_thresh,
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

        class_id = infer_cell(self.str_tree, cell, ioa_thresh,
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

        class_id = infer_cell(self.str_tree, cell, ioa_thresh,
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

        class_id = infer_cell(self.str_tree, cell, ioa_thresh,
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

        class_id = infer_cell(self.str_tree, cell, ioa_thresh,
                              use_intersection_over_cell, background_class_id,
                              pick_min_class_id)
        self.assertEqual(class_id, self.class_id2)

    def test_infer_labels(self):
        extent = Box.make_square(0, 0, 4)
        options = ClassificationGeoJSONFileConfig.Options()
        options.ioa_thresh = 0.5
        options.use_intersection_over_cell = False
        options.background_class_id = self.background_class_id
        options.pick_min_class_id = False
        options.infer_cells = True
        options.cell_size = 2

        labels = infer_labels(self.geojson_dict, self.crs_transformer, extent,
                              options)
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
        extent = Box.make_square(0, 0, 2.5)
        labels = read_labels(self.geojson_dict, self.crs_transformer, extent)

        cells = labels.get_cells()
        self.assertEqual(len(cells), 1)
        class_id = labels.get_cell_class_id(self.box1)
        self.assertEqual(class_id, self.class_id1)
        class_id = labels.get_cell_class_id(self.box2)
        self.assertEqual(class_id, None)

    def test_read_labels2(self):
        # Extent contains both boxes.
        extent = Box.make_square(0, 0, 4)
        labels = read_labels(self.geojson_dict, self.crs_transformer, extent)

        cells = labels.get_cells()
        self.assertEqual(len(cells), 2)
        class_id = labels.get_cell_class_id(self.box1)
        self.assertEqual(class_id, self.class_id1)
        class_id = labels.get_cell_class_id(self.box2)
        self.assertEqual(class_id, self.class_id2)

    def test_to_geojson(self):
        extent = Box.make_square(0, 0, 4)
        labels = read_labels(self.geojson_dict, self.crs_transformer, extent)
        geojson_dict = to_geojson(labels, self.crs_transformer, self.class_map)
        self.assertDictEqual(geojson_dict, self.geojson_dict)

    def test_constructor_save(self):
        # Read it, write it using label_store, read it again, and compare.
        extent = Box.make_square(0, 0, 10)

        options = ClassificationGeoJSONFileConfig.Options()
        options.infer_cells = False

        label_store = ClassificationGeoJSONFile(
            self.file_path,
            self.aoi_file_path,
            self.crs_transformer,
            options,
            self.class_map,
            extent,
            readable=True,
            writable=True)
        labels1 = label_store.get_labels()
        label_store.save()

        label_store = ClassificationGeoJSONFile(
            self.file_path,
            self.aoi_file_path,
            self.crs_transformer,
            options,
            self.class_map,
            extent=None,
            readable=True,
            writable=True)
        labels2 = label_store.get_labels()

        self.assertDictEqual(labels1.cell_to_class_id,
                             labels2.cell_to_class_id)

    def test_constructor_aoi(self):
        extent = Box.make_square(0, 0, 10)
        options = ClassificationGeoJSONFileConfig.Options()
        options.infer_cells = False
        label_store = ClassificationGeoJSONFile(
            self.file_path,
            self.aoi_file_path,
            self.crs_transformer,
            options,
            self.class_map,
            extent,
            readable=True,
            writable=True)
        aoi = [Box.make_square(0, 0, 2).get_shapely()]
        self.assertEqual(aoi, label_store.aoi)


if __name__ == '__main__':
    unittest.main()
