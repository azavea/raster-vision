from typing import Callable
from os.path import join
import unittest

from rastervision.pipeline.file_system.utils import get_tmp_dir, file_exists
from rastervision.core.data import (ClassConfig, IdentityCRSTransformer,
                                    ObjectDetectionGeoJSONStore,
                                    ObjectDetectionGeoJSONStoreConfig)

from tests import data_file_path


class TestObjectDetectionGeoJSONStoreConfig(unittest.TestCase):
    def test_build(self):
        uri = data_file_path('bboxes.geojson')
        class_config = ClassConfig(names=['1', '2'])
        crs_transformer = IdentityCRSTransformer()
        cfg = ObjectDetectionGeoJSONStoreConfig(uri=uri)
        ls = cfg.build(
            class_config=class_config, crs_transformer=crs_transformer)
        self.assertIsInstance(ls, ObjectDetectionGeoJSONStore)


class TestObjectDetectionGeoJSONStore(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_get_labels(self):
        uri = data_file_path('bboxes.geojson')
        class_config = ClassConfig(names=['1', '2'])
        crs_transformer = IdentityCRSTransformer()
        ls = ObjectDetectionGeoJSONStore(
            uri,
            class_config=class_config,
            crs_transformer=crs_transformer,
        )
        self.assertNoError(lambda: ls.get_labels())

    def test_save(self):
        uri = data_file_path('bboxes.geojson')
        class_config = ClassConfig(names=['1', '2'])
        crs_transformer = IdentityCRSTransformer()
        ls = ObjectDetectionGeoJSONStore(
            uri,
            class_config=class_config,
            crs_transformer=crs_transformer,
        )
        labels = ls.get_labels()
        with get_tmp_dir() as tmp_dir:
            save_uri = join(tmp_dir, 'labels.geojson')
            ls.uri = save_uri
            ls.save(labels)
            self.assertTrue(file_exists(save_uri))


if __name__ == '__main__':
    unittest.main()
