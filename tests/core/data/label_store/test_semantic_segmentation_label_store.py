from typing import Callable
from os.path import join
import unittest

import numpy as np

from rastervision.pipeline.file_system.utils import get_tmp_dir, file_exists
from rastervision.core.box import Box
from rastervision.core.data import (
    BuildingVectorOutputConfig, ClassConfig, IdentityCRSTransformer,
    PolygonVectorOutputConfig, SemanticSegmentationLabelStore,
    SemanticSegmentationSmoothLabels, VectorOutputConfig)
from tests.core.data.label.test_semantic_segmentation_labels import (
    make_random_scores)


class TestVectorOutputConfig(unittest.TestCase):
    def test_get_uri(self):
        # w/o ClasConfig
        cfg = VectorOutputConfig(class_id=0)
        self.assertEqual(cfg.get_uri('abc/def'), 'abc/def/class-0.json')

        # w/ ClasConfig
        class_config = ClassConfig(names=['a', 'b'])
        cfg = VectorOutputConfig(class_id=0)
        self.assertEqual(
            cfg.get_uri('abc/def', class_config), 'abc/def/class-0-a.json')
        cfg = VectorOutputConfig(class_id=1)
        self.assertEqual(
            cfg.get_uri('abc/def', class_config), 'abc/def/class-1-b.json')


class TestPolygonVectorOutputConfig(unittest.TestCase):
    def test_denoise(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[60:65, 60:65] = 1

        # denoise = 0
        cfg = PolygonVectorOutputConfig(class_id=0, denoise=0)
        polys = list(cfg.vectorize(mask))
        self.assertEqual(len(polys), 2)

        # denoise = 8
        cfg = PolygonVectorOutputConfig(class_id=0, denoise=8)
        polys = list(cfg.vectorize(mask))
        self.assertEqual(len(polys), 1)


class TestBuildingVectorOutputConfig(unittest.TestCase):
    def test_denoise(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[60:65, 60:65] = 1

        # denoise = 0
        cfg = BuildingVectorOutputConfig(class_id=0, denoise=0)
        polys = list(cfg.vectorize(mask))
        self.assertEqual(len(polys), 2)

        # denoise = 8
        cfg = BuildingVectorOutputConfig(class_id=0, denoise=8)
        polys = list(cfg.vectorize(mask))
        self.assertEqual(len(polys), 1)


class TestSemanticSegmentationLabelStore(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_saving_and_loading(self):
        with get_tmp_dir() as tmp_dir:
            class_config = ClassConfig(names=['bg', 'fg'], null_class='bg')
            label_store = SemanticSegmentationLabelStore(
                uri=tmp_dir,
                crs_transformer=IdentityCRSTransformer(),
                class_config=class_config,
                bbox=None,
                smooth_output=True,
                smooth_as_uint8=True,
                vector_outputs=[PolygonVectorOutputConfig(class_id=1)])
            labels = SemanticSegmentationSmoothLabels(
                extent=Box(0, 0, 10, 10), num_classes=len(class_config))
            labels.pixel_scores += make_random_scores(
                len(class_config), 10, 10)
            labels.pixel_hits += 1
            label_store.save(labels)

            self.assertTrue(file_exists(join(tmp_dir, 'labels.tif')))
            self.assertTrue(file_exists(join(tmp_dir, 'scores.tif')))
            self.assertTrue(file_exists(join(tmp_dir, 'pixel_hits.npy')))

            del label_store

            # test compatibility validation
            args = dict(
                uri=tmp_dir,
                crs_transformer=IdentityCRSTransformer(),
                class_config=ClassConfig(names=['bg', 'fg', 'null']),
                smooth_output=True,
                smooth_as_uint8=True,
            )
            with self.assertRaises(FileExistsError):
                label_store = SemanticSegmentationLabelStore(**args)

            args = dict(
                uri=tmp_dir,
                crs_transformer=IdentityCRSTransformer(),
                class_config=class_config,
                smooth_output=True,
                smooth_as_uint8=True,
            )
            label_store = SemanticSegmentationLabelStore(**args)
            self.assertIsNotNone(label_store.label_source)
            self.assertIsNotNone(label_store.score_source)

            self.assertNoError(lambda: label_store.get_labels())
            self.assertNoError(lambda: label_store.get_scores())

            self.assertNoError(lambda: label_store.save(labels))


if __name__ == '__main__':
    unittest.main()
