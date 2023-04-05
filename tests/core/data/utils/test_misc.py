from typing import Any, Tuple
import unittest
from os.path import join

from rastervision.pipeline.file_system.utils import get_tmp_dir, json_to_file
from rastervision.core.box import Box
from rastervision.core.data import (
    ClassConfig, GeoJSONVectorSource, RasterioSource,
    ChipClassificationLabelSource, ChipClassificationLabelSourceConfig,
    ChipClassificationGeoJSONStore, ObjectDetectionLabelSource,
    ObjectDetectionGeoJSONStore, SemanticSegmentationLabelSource,
    SemanticSegmentationLabelStore)
from rastervision.core.data.utils.geojson import geoms_to_geojson
from rastervision.core.data.utils.misc import (match_extents,
                                               parse_array_slices)

from tests import data_file_path


class TestMatchExtents(unittest.TestCase):
    def setUp(self) -> None:
        self.class_config = ClassConfig(names=['class_1'])
        self.rs_path = data_file_path(
            'multi_raster_source/const_100_600x600.tiff')
        self.extent_rs = Box(4, 4, 8, 8)
        self.raster_source = RasterioSource(
            self.rs_path, extent=self.extent_rs)
        self.crs_tf = self.raster_source.crs_transformer

        self.extent_ls = Box(0, 0, 12, 12)
        geoms = [b.to_shapely() for b in self.extent_ls.get_windows(2, 2)]
        geoms = [self.crs_tf.pixel_to_map(g) for g in geoms]
        properties = [dict(class_id=0) for _ in geoms]
        geojson = geoms_to_geojson(geoms, properties)
        self._tmp_dir = get_tmp_dir()
        self.tmp_dir = self._tmp_dir.name
        uri = join(self.tmp_dir, 'labels.json')
        json_to_file(geojson, uri)
        self.vector_source = GeoJSONVectorSource(
            uri, self.raster_source.crs_transformer, ignore_crs_field=True)

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def test_cc_label_source(self):
        label_source = ChipClassificationLabelSource(
            ChipClassificationLabelSourceConfig(),
            vector_source=self.vector_source)
        self.assertEqual(label_source.extent, self.extent_ls)
        match_extents(self.raster_source, label_source)
        self.assertEqual(label_source.extent, self.raster_source.extent)

    def test_cc_label_store(self):
        uri = join(self.tmp_dir, 'cc_labels.json')
        label_store = ChipClassificationGeoJSONStore(uri, self.class_config,
                                                     self.crs_tf)
        self.assertIsNone(label_store.extent)
        match_extents(self.raster_source, label_store)
        self.assertEqual(label_store.extent, self.raster_source.extent)

    def test_od_label_source(self):
        label_source = ObjectDetectionLabelSource(
            vector_source=self.vector_source)
        self.assertEqual(label_source.extent, self.extent_ls)
        match_extents(self.raster_source, label_source)
        self.assertEqual(label_source.extent, self.raster_source.extent)

    def test_od_label_store(self):
        uri = join(self.tmp_dir, 'od_labels.json')
        label_store = ObjectDetectionGeoJSONStore(uri, self.class_config,
                                                  self.crs_tf)
        self.assertIsNone(label_store.extent)
        match_extents(self.raster_source, label_store)
        self.assertEqual(label_store.extent, self.raster_source.extent)

    def test_ss_label_source(self):
        label_source = SemanticSegmentationLabelSource(
            self.raster_source, class_config=self.class_config)
        self.assertEqual(label_source.extent, self.extent_rs)
        match_extents(self.raster_source, label_source)
        self.assertEqual(label_source.extent, self.raster_source.extent)

    def test_ss_label_store(self):
        uri = join(self.tmp_dir, 'ss_labels')
        label_store = SemanticSegmentationLabelStore(
            uri,
            extent=self.extent_ls,
            crs_transformer=self.crs_tf,
            class_config=self.class_config)
        self.assertEqual(label_store.extent, self.extent_ls)
        match_extents(self.raster_source, label_store)
        self.assertEqual(label_store.extent, self.raster_source.extent)


class TestParseArraySlices(unittest.TestCase):
    class MockSource:
        def __init__(self, dims: int, extent: Box) -> None:
            self.dims = dims
            self.extent = extent

        def __getitem__(self, key: Any) -> Tuple[Box, list]:
            return parse_array_slices(key, self.extent, dims=self.dims)

    def test_errors(self):
        source = self.MockSource(dims=3, extent=Box(0, 0, 100, 100))
        self.assertRaises(TypeError, lambda: source['a'])
        self.assertRaises(IndexError, lambda: source[:10, :10, 0, 0])
        self.assertRaises(TypeError, lambda: source[:10, :10, None])
        self.assertRaises(ValueError, lambda: source[10, :10])
        self.assertRaises(NotImplementedError, lambda: source[:-10, :10])
        self.assertRaises(NotImplementedError, lambda: source[:10, :-10])
        self.assertRaises(NotImplementedError, lambda: source[::-1])
        self.assertRaises(NotImplementedError, lambda: source[:, ::-1])

    def test_window(self):
        source = self.MockSource(dims=2, extent=Box(0, 0, 100, 100))

        window, _ = source[5:10, 15:20]
        self.assertEqual(window, Box(5, 15, 10, 20))

        window, _ = source[5:10, :]
        self.assertEqual(window, Box(5, 0, 10, 100))

        window, _ = source[:, 15:20]
        self.assertEqual(window, Box(0, 15, 100, 20))

        window, _ = source[5:10]
        self.assertEqual(window, Box(5, 0, 10, 100))

    def test_dim_slices(self):
        source = self.MockSource(dims=3, extent=Box(0, 0, 100, 100))

        _, dim_slices = source[5:10, 15:20]
        self.assertListEqual(dim_slices, [slice(5, 10), slice(15, 20), None])

        _, dim_slices = source[5:10, 15:20, 0]
        self.assertListEqual(dim_slices, [slice(5, 10), slice(15, 20), 0])

        _, dim_slices = source[5:10, 15:20, 1:4]
        self.assertListEqual(
            dim_slices,
            [slice(5, 10), slice(15, 20),
             slice(1, 4)])

        _, dim_slices = source[5:10, 15:20, [3, 1]]
        self.assertListEqual(dim_slices, [slice(5, 10), slice(15, 20), [3, 1]])

        source = self.MockSource(dims=4, extent=Box(0, 0, 100, 100))
        _, dim_slices = source[5:10, 15:20, 0]
        self.assertListEqual(
            dim_slices, [slice(5, 10), slice(15, 20), 0, None])

    def test_ellipsis(self):
        source = self.MockSource(dims=3, extent=Box(0, 0, 100, 100))

        window, dim_slices = source[5:10, 15:20, ...]
        self.assertEqual(window, Box(5, 15, 10, 20))
        self.assertListEqual(dim_slices, [slice(5, 10), slice(15, 20), None])

        window, dim_slices = source[5:10, ...]
        self.assertEqual(window, Box(5, 0, 10, 100))
        self.assertListEqual(dim_slices, [slice(5, 10), slice(0, 100), None])

        window, dim_slices = source[5:10, ..., 0]
        self.assertEqual(window, Box(5, 0, 10, 100))
        self.assertListEqual(dim_slices, [slice(5, 10), slice(0, 100), 0])

        window, dim_slices = source[..., 15:20, 0]
        self.assertEqual(window, Box(0, 15, 100, 20))
        self.assertListEqual(dim_slices, [slice(0, 100), slice(15, 20), 0])

        window, dim_slices = source[..., 0]
        self.assertEqual(window, Box(0, 0, 100, 100))
        self.assertListEqual(dim_slices, [slice(0, 100), slice(0, 100), 0])

    def test_cropped_extent(self):
        source = self.MockSource(dims=2, extent=Box(20, 30, 80, 70))

        window, dim_slices = source[5:10, 15:20]
        self.assertEqual(window, Box(5, 15, 10, 20))
        self.assertListEqual(dim_slices, [slice(5, 10), slice(15, 20)])

        window, dim_slices = source[:, :]
        self.assertEqual(window, Box(0, 0, 60, 40))
        self.assertListEqual(dim_slices, [slice(0, 60), slice(0, 40)])

        window, dim_slices = source[:]
        self.assertEqual(window, Box(0, 0, 60, 40))
        self.assertListEqual(dim_slices, [slice(0, 60), slice(0, 40)])

        window, dim_slices = source[..., :]
        self.assertEqual(window, Box(0, 0, 60, 40))
        self.assertListEqual(dim_slices, [slice(0, 60), slice(0, 40)])

    def test_step(self):
        source = self.MockSource(dims=2, extent=Box(20, 30, 80, 70))

        window, dim_slices = source[5:10:2, 15:20:3]
        self.assertEqual(window, Box(5, 15, 10, 20))
        self.assertListEqual(dim_slices, [slice(5, 10, 2), slice(15, 20, 3)])

        window, dim_slices = source[::2, ::3]
        self.assertEqual(window, Box(0, 0, 60, 40))
        self.assertListEqual(dim_slices, [slice(0, 60, 2), slice(0, 40, 3)])


if __name__ == '__main__':
    unittest.main()
