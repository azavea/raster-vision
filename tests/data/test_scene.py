import unittest

import rastervision as rv
from rastervision.rv_config import RVConfig

import tests.mock as mk
from tests import data_file_path


class TestScene(unittest.TestCase):
    def setUp(self):
        config = {'PLUGINS_modules': '["{}"]'.format('tests.mock')}
        rv._registry.initialize_config(config_overrides=config)

        self.temp_dir = RVConfig.get_tmp_dir()

    def tearDown(self):
        rv._registry.initialize_config()
        self.temp_dir.cleanup()

    def test_with_aois(self):
        aoi_uri = data_file_path('evaluator/cc-label-aoi.json')
        aoi_uris = [aoi_uri, aoi_uri]
        task_config = rv.TaskConfig.builder(mk.MOCK_TASK).build()

        scene_config = mk.create_mock_scene()
        scene_config = scene_config.to_builder().with_aoi_uris(
            aoi_uris).build()
        scene = scene_config.create_scene(task_config, self.temp_dir.name)
        self.assertEqual(2, len(scene.aoi_polygons))

    def test_with_aoi(self):
        aoi_uri = data_file_path('evaluator/cc-label-aoi.json')
        task_config = rv.TaskConfig.builder(mk.MOCK_TASK).build()

        scene_config = mk.create_mock_scene()
        scene_config = scene_config.to_builder().with_aoi_uri(aoi_uri).build()
        scene = scene_config.create_scene(task_config, self.temp_dir.name)
        self.assertEqual(1, len(scene.aoi_polygons))

    def test_with_aoi_back_compat(self):
        aoi_uri = data_file_path('evaluator/cc-label-aoi.json')
        task_config = rv.TaskConfig.builder(mk.MOCK_TASK).build()
        scene_config = mk.create_mock_scene()
        scene_msg = scene_config.to_proto()
        # Use deprecated aoi_uri field.
        del scene_msg.aoi_uris[:]
        scene_msg.aoi_uri = aoi_uri
        scene_config = rv.SceneConfig.from_proto(scene_msg)
        scene = scene_config.create_scene(task_config, self.temp_dir.name)
        self.assertEqual(1, len(scene.aoi_polygons))


if __name__ == '__main__':
    unittest.main()
