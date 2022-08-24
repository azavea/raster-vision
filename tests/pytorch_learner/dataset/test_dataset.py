from os.path import join
import unittest

import torch
import numpy as np
from shapely.geometry import Polygon, mapping

from rastervision.pipeline import rv_config
from rastervision.pipeline.file_system import json_to_file
from rastervision.core.data import ClassConfig, RasterioCRSTransformer
from rastervision.core.data.utils.geojson import (geometry_to_feature,
                                                  features_to_geojson)
from rastervision.pytorch_learner.dataset import (
    SemanticSegmentationSlidingWindowGeoDataset,
    ClassificationSlidingWindowGeoDataset,
    ObjectDetectionSlidingWindowGeoDataset)

from tests import data_file_path


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
        self.tmp_dir = rv_config.get_tmp_dir()
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
            class_config=class_config, image_uri=image_uri, size=10, stride=10)
        x, y = ds[0]
        torch.testing.assert_allclose(x * 255, torch.ones_like(x))
        self.assertTrue(np.isnan(y.numpy()))

        # raster labels
        ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
            class_config=class_config,
            image_uri=image_uri,
            label_raster_uri=image_uri,
            size=10,
            stride=10)
        x, y = ds[0]
        torch.testing.assert_allclose(x * 255, torch.ones_like(x))
        self.assertAlmostEqual(y.float().mean(), 1)

        # rasterized labels
        ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
            class_config=class_config,
            image_uri=image_uri,
            label_vector_uri=label_vector_uri,
            size=10,
            stride=10)
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
            class_config=class_config, image_uri=image_uri, size=10, stride=10)
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
            stride=10)
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
            class_config=class_config, image_uri=image_uri, size=10, stride=10)
        x, y = ds[0]
        torch.testing.assert_allclose(x * 255, torch.ones_like(x))
        self.assertTrue(np.isnan(y.numpy()))

        # vector labels
        ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
            class_config=class_config,
            image_uri=image_uri,
            label_vector_uri=label_vector_uri,
            size=20,
            stride=20)
        x, y = ds[0]
        bboxes, class_ids, _ = y
        np.testing.assert_allclose(bboxes, np.array([[0., 0., 10., 10.]]))
        np.testing.assert_allclose(class_ids, np.array([1]))
        x, y = ds[1]
        bboxes, class_ids, _ = y
        self.assertTupleEqual(bboxes.shape, (0, 4))
        self.assertTupleEqual(class_ids.shape, (0, ))


if __name__ == '__main__':
    unittest.main()
