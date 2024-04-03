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

from tests import data_file_path


class MockScene:
    aoi_polygons_bbox_coords: list['Polygon'] = [
        Box(0, 0, 10, 10).to_shapely()
    ]
    extent = Box(0, 0, 10, 10)


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
            (12, 13),
            within_aoi=True,
            transform_type=TransformType.noop,
        )
        self.assertRaises(StopIteration, lambda: ds.sample_window())

        ds = RandomWindowGeoDataset(
            scene,
            10,
            (12, 13),
            within_aoi=False,
            transform_type=TransformType.noop,
        )
        self.assertNoError(StopIteration, lambda: ds.sample_window())


if __name__ == '__main__':
    unittest.main()
