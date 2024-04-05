from typing import Callable
from os.path import join
import unittest

import torch
import numpy as np
from shapely.geometry import Polygon, mapping

from rastervision.pipeline.file_system import json_to_file, get_tmp_dir
from rastervision.core.box import Box
from rastervision.core.data import ClassConfig, RasterioCRSTransformer
from rastervision.core.data.utils.geojson import (geometry_to_feature,
                                                  features_to_geojson)
from rastervision.pytorch_learner.dataset import (
    SemanticSegmentationSlidingWindowGeoDataset,
    ClassificationSlidingWindowGeoDataset,
    ObjectDetectionSlidingWindowGeoDataset, RandomWindowGeoDataset,
    SlidingWindowGeoDataset, TransformType)
from rastervision.pytorch_learner.dataset.dataset import _to_tuple

from tests import data_file_path


class MockScene:
    aoi_polygons_bbox_coords: list['Polygon'] = [
        Box(0, 0, 10, 10).to_shapely()
    ]
    extent = Box(0, 0, 10, 10)

    def __getitem__(self, key):
        return np.empty((1, 1, 3)), np.empty((1, 1))


def make_overlapping_geojson(uri: str) -> str:
    crs_tf = RasterioCRSTransformer.from_uri(uri)
    xmin, ymin = crs_tf.pixel_to_map((0, 0))
    xmax, ymax = crs_tf.pixel_to_map((10, 10))
    polygon = Polygon.from_bounds(xmin, ymin, xmax, ymax)
    geometry = mapping(polygon)
    feature = geometry_to_feature(geometry, properties=dict(class_id=1))
    geojson = features_to_geojson([feature])
    return geojson


class TestGeoDatasetFromURIs(unittest.TestCase):
    def setUp(self) -> None:
        self.image_uri = data_file_path('ones.tif')
        self.tmp_dir = get_tmp_dir()
        geojson = make_overlapping_geojson(self.image_uri)
        self.label_vector_uri = join(self.tmp_dir.name, 'geojson.json')
        json_to_file(geojson, self.label_vector_uri)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_ss_from_uris(self):
        class_config = ClassConfig(names=['bg', 'fg'], null_class='bg')
        image_uri = self.image_uri
        label_vector_uri = self.label_vector_uri

        # no labels
        ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
            image_uri=image_uri, size=10, stride=10, padding=0)
        x, y = ds[0]
        torch.testing.assert_allclose(x * 255, torch.ones_like(x))
        self.assertTrue(np.isnan(y.numpy()))

        # raster labels
        ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
            class_config=class_config,
            image_uri=image_uri,
            label_raster_uri=image_uri,
            size=10,
            stride=10,
            padding=0)
        x, y = ds[0]
        torch.testing.assert_allclose(x * 255, torch.ones_like(x))
        self.assertAlmostEqual(y.float().mean(), 1)

        # rasterized labels
        ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
            class_config=class_config,
            image_uri=image_uri,
            label_vector_uri=label_vector_uri,
            size=10,
            stride=10,
            padding=0)
        x, y = ds[0]
        torch.testing.assert_allclose(x * 255, torch.ones_like(x))
        torch.testing.assert_allclose(y, torch.ones_like(y))
        x, y = ds[3]
        torch.testing.assert_allclose(x * 255, torch.ones_like(x))
        torch.testing.assert_allclose(y, torch.zeros_like(y))

    def test_cc_from_uris(self):
        class_config = ClassConfig(names=['bg', 'fg'], null_class='bg')
        image_uri = self.image_uri
        label_vector_uri = self.label_vector_uri

        # no labels
        ds = ClassificationSlidingWindowGeoDataset.from_uris(
            image_uri=image_uri, size=10, stride=10, padding=0)
        x, y = ds[0]
        torch.testing.assert_allclose(x * 255, torch.ones_like(x))
        self.assertTrue(np.isnan(y.numpy()))

        # vector labels
        ds = ClassificationSlidingWindowGeoDataset.from_uris(
            class_config=class_config,
            image_uri=image_uri,
            label_vector_uri=label_vector_uri,
            label_source_kw=dict(background_class_id=0),
            size=10,
            stride=10,
            padding=0)
        x, y = ds[0]
        torch.testing.assert_allclose(x * 255, torch.ones_like(x))
        torch.testing.assert_allclose(y, torch.ones_like(y))
        x, y = ds[3]
        torch.testing.assert_allclose(x * 255, torch.ones_like(x))
        torch.testing.assert_allclose(y, torch.zeros_like(y))

    def test_od_from_uris(self):
        class_config = ClassConfig(names=['bg', 'fg'], null_class='bg')
        image_uri = self.image_uri
        label_vector_uri = self.label_vector_uri

        # no labels
        ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
            image_uri=image_uri, size=10, stride=10, padding=0)
        x, y = ds[0]
        torch.testing.assert_allclose(x * 255, torch.ones_like(x))
        self.assertTrue(np.isnan(y.numpy()))

        # vector labels
        ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
            class_config=class_config,
            image_uri=image_uri,
            label_vector_uri=label_vector_uri,
            size=20,
            stride=20,
            padding=0)
        x, y = ds[0]
        bboxes = y.get_field('boxes')
        class_ids = y.get_field('class_ids')
        np.testing.assert_allclose(bboxes, np.array([[0., 0., 10., 10.]]))
        np.testing.assert_allclose(class_ids, np.array([1]))

        x, y = ds[1]
        bboxes = y.get_field('boxes')
        class_ids = y.get_field('class_ids')
        self.assertTupleEqual(bboxes.shape, (0, 4))
        self.assertTupleEqual(class_ids.shape, (0, ))


class TestSlidingWindowGeoDataset(unittest.TestCase):
    def test_sample_window_within_aoi(self):
        scene = MockScene()
        ds = SlidingWindowGeoDataset(
            scene,
            10,
            5,
            within_aoi=True,
            transform_type=TransformType.noop,
        )
        self.assertEqual(len(ds.windows), 1)

        ds = SlidingWindowGeoDataset(
            scene,
            10,
            5,
            within_aoi=False,
            transform_type=TransformType.noop,
        )
        self.assertEqual(len(ds.windows), 4)

    def test_out_size(self):
        scene = MockScene()
        ds = SlidingWindowGeoDataset(
            scene,
            size=5,
            stride=5,
            out_size=10,
            transform_type=TransformType.semantic_segmentation,
        )
        x, y = ds[0]
        self.assertTupleEqual(x.shape, (3, 10, 10))
        self.assertTupleEqual(y.shape, (10, 10))

    def test_return_window(self):
        scene = MockScene()
        ds = SlidingWindowGeoDataset(
            scene,
            10,
            5,
            transform_type=TransformType.noop,
            return_window=True,
        )
        out = ds[0]
        self.assertEqual(len(out), 2)
        _, window = out
        self.assertIsInstance(window, Box)


class TestRandomWindowGeoDataset(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_sample_window_within_aoi(self):
        scene = MockScene()

        ds = RandomWindowGeoDataset(
            scene,
            10,
            (5, 6),
            within_aoi=True,
            transform_type=TransformType.noop,
        )
        self.assertNoError(ds.sample_window)

        ds = RandomWindowGeoDataset(
            scene,
            10,
            (12, 13),
            within_aoi=True,
            transform_type=TransformType.noop,
        )
        self.assertRaises(StopIteration, ds.sample_window)

        ds = RandomWindowGeoDataset(
            scene,
            10,
            (12, 13),
            within_aoi=False,
            transform_type=TransformType.noop,
        )
        self.assertNoError(ds.sample_window)

    def test_init_validation(self):
        scene = MockScene()

        # neither size_lims or h/w_lims specified
        args = dict(
            scene=scene,
            out_size=10,
            transform_type=TransformType.noop,
        )
        self.assertRaises(ValueError, lambda: RandomWindowGeoDataset(**args))

        # size_lims + h_lims specified
        args = dict(
            scene=scene,
            out_size=10,
            size_lims=(10, 11),
            h_lims=(10, 11),
            transform_type=TransformType.noop,
        )
        self.assertRaises(ValueError, lambda: RandomWindowGeoDataset(**args))

        # size_lims + h_lims + w_lims specified
        args = dict(
            scene=scene,
            out_size=10,
            size_lims=(10, 11),
            h_lims=(10, 11),
            w_lims=(10, 11),
            transform_type=TransformType.noop,
        )
        self.assertRaises(ValueError, lambda: RandomWindowGeoDataset(**args))

        # only w_lims specified
        args = dict(
            scene=scene,
            out_size=10,
            w_lims=(10, 11),
            transform_type=TransformType.noop,
        )
        self.assertRaises(ValueError, lambda: RandomWindowGeoDataset(**args))

        # out_size=None
        ds = RandomWindowGeoDataset(
            scene,
            out_size=None,
            size_lims=(12, 13),
            transform_type=TransformType.noop,
        )
        self.assertFalse(ds.normalize)
        self.assertFalse(ds.to_pytorch)

        # padding initialization
        ds = RandomWindowGeoDataset(
            scene,
            out_size=None,
            h_lims=(10, 11),
            w_lims=(10, 11),
            transform_type=TransformType.noop,
        )
        self.assertTupleEqual(ds.padding, (5, 5))

    def test_min_max_size(self):
        scene = MockScene()
        ds = RandomWindowGeoDataset(
            scene,
            out_size=None,
            size_lims=(10, 15),
            transform_type=TransformType.noop,
        )
        self.assertTupleEqual(ds.min_size, (10, 10))
        self.assertTupleEqual(ds.max_size, (15, 15))

        ds = RandomWindowGeoDataset(
            scene,
            out_size=None,
            h_lims=(10, 15),
            w_lims=(8, 12),
            transform_type=TransformType.noop,
        )
        self.assertTupleEqual(ds.min_size, (10, 8))
        self.assertTupleEqual(ds.max_size, (15, 12))

    def test_sample_window_size(self):
        scene = MockScene()
        ds = RandomWindowGeoDataset(
            scene,
            out_size=None,
            size_lims=(10, 15),
            transform_type=TransformType.noop,
        )
        sampled_h, sampled_w = ds.sample_window_size()
        self.assertTrue(10 <= sampled_h < 15)
        self.assertTrue(10 <= sampled_w < 15)

        ds = RandomWindowGeoDataset(
            scene,
            out_size=None,
            h_lims=(10, 15),
            w_lims=(8, 12),
            transform_type=TransformType.noop,
        )
        sampled_h, sampled_w = ds.sample_window_size()
        self.assertTrue(10 <= sampled_h < 15)
        self.assertTrue(8 <= sampled_w < 12)

    def test_max_windows(self):
        scene = MockScene()
        ds = RandomWindowGeoDataset(
            scene,
            out_size=10,
            size_lims=(10, 11),
            max_windows=10,
            transform_type=TransformType.noop,
        )
        self.assertRaises(StopIteration, lambda: ds[10])

    def test_return_window(self):
        scene = MockScene()
        ds = RandomWindowGeoDataset(
            scene,
            out_size=10,
            size_lims=(5, 6),
            transform_type=TransformType.noop,
            return_window=True,
        )
        out = ds[0]
        self.assertEqual(len(out), 2)
        _, window = out
        self.assertIsInstance(window, Box)

    def test_triangle_missing(self):
        import sys
        sys.modules['triangle'] = None
        scene = MockScene()
        args = dict(
            scene=scene,
            out_size=10,
            size_lims=(5, 6),
            transform_type=TransformType.noop,
        )
        self.assertNoError(lambda: RandomWindowGeoDataset(**args))
        ds = RandomWindowGeoDataset(**args)
        self.assertIsNone(ds.aoi_sampler)


class TestUtils(unittest.TestCase):
    def test__to_tuple(self):
        self.assertTupleEqual(_to_tuple(1, 2), (1, 1))
        self.assertRaises(ValueError, lambda: _to_tuple((1, 1, 1), 2))


if __name__ == '__main__':
    unittest.main()
