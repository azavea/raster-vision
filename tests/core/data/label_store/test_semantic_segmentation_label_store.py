import unittest
from os.path import realpath

from rastervision.core.data.label_store import (
    PolygonVectorOutputConfig, BuildingVectorOutputConfig,
    SemanticSegmentationLabelStoreConfig)


class MockPipelineConfig():
    predict_uri = '/abc/def'


class MockSceneConfig():
    id = 'scene-0'


class TestSemanticSegmentationLabelStoreConfig(unittest.TestCase):
    def test_vector_output_config(self):
        # uri not updated if no pipeline and scene
        cfg = PolygonVectorOutputConfig(class_id=0)
        cfg.update()
        self.assertIsNone(cfg.uri)

        cfg = BuildingVectorOutputConfig(class_id=0)
        cfg.update()
        self.assertIsNone(cfg.uri)

    def test_polygon_vector_output_config(self):
        cfg = PolygonVectorOutputConfig(class_id=1)
        cfg.update(pipeline=MockPipelineConfig(), scene=MockSceneConfig())

        # correct mode
        self.assertEqual(cfg.get_mode(), 'polygons')
        # uri updated based on pipeline and scene
        self.assertEqual(cfg.uri,
                         '/abc/def/scene-0/vector_output/polygons-1.json')

    def test_building_vector_output_config(self):
        cfg = BuildingVectorOutputConfig(class_id=2)
        cfg.update(pipeline=MockPipelineConfig(), scene=MockSceneConfig())

        # correct mode
        self.assertEqual(cfg.get_mode(), 'buildings')
        # uri updated based on pipeline and scene
        self.assertEqual(cfg.uri,
                         '/abc/def/scene-0/vector_output/buildings-2.json')

    def test_semantic_segmentation_label_store_config(self):
        # uri not updated if no pipeline and scene
        cfg = SemanticSegmentationLabelStoreConfig()
        cfg.update()
        self.assertIsNone(cfg.uri)

        cfg = SemanticSegmentationLabelStoreConfig(vector_output=[
            PolygonVectorOutputConfig(class_id=1),
            BuildingVectorOutputConfig(class_id=2)
        ])
        cfg.update(pipeline=MockPipelineConfig(), scene=MockSceneConfig())
        # uri updated based on pipeline and scene
        self.assertEqual(realpath(cfg.uri), '/abc/def/scene-0')

        # vector outputs updated
        self.assertEqual(cfg.vector_output[0].uri,
                         '/abc/def/scene-0/vector_output/polygons-1.json')
        self.assertEqual(cfg.vector_output[1].uri,
                         '/abc/def/scene-0/vector_output/buildings-2.json')


if __name__ == '__main__':
    unittest.main()
