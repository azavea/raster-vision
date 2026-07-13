import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from rastervision.pipeline.config import Config, Field
from rastervision.pipeline.file_system.utils import collect_uris
from rastervision.pipeline.cli import main


# Mock configs for testing collect_uris
class MockRasterSourceConfig(Config):
    uris: list[str] = Field(default_factory=lambda: ['/path/to/img.tif'])
    channel_order: list[int] | None = None


class MockVectorSourceConfig(Config):
    uris: str | list[str] = Field(default_factory=lambda: '/path/to/data.geojson')


class MockSceneConfig(Config):
    id: str = 'scene1'
    raster_source: MockRasterSourceConfig = Field(
        default_factory=MockRasterSourceConfig)
    vector_source: MockVectorSourceConfig | None = None
    aoi_uris: list[str] | None = None


class MockDatasetConfig(Config):
    train_scenes: list[MockSceneConfig] = Field(
        default_factory=lambda: [MockSceneConfig()])
    validation_scenes: list[MockSceneConfig] = Field(default_factory=list)


class MockPipelineConfig(Config):
    dataset: MockDatasetConfig | None = None
    root_uri: str | None = None


class MockSTACItemConfig(Config):
    uri: str = Field(default='https://example.com/stac/item.json')


class MockSTACCollectionConfig(Config):
    uri: str = Field(default='https://example.com/stac/collection.json')


class TestCollectUris(unittest.TestCase):
    """Tests for the collect_uris utility function."""

    def test_simple_uri_field(self):
        """Test collecting a single str URI field."""
        cfg = MockSTACItemConfig()
        result = collect_uris(cfg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 'uri')
        self.assertEqual(result[0][1], 'https://example.com/stac/item.json')

    def test_uris_list_field(self):
        """Test collecting a uris: list[str] field."""
        raster_cfg = MockRasterSourceConfig()
        result = collect_uris(raster_cfg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 'uris[0]')
        self.assertEqual(result[0][1], '/path/to/img.tif')

    def test_uris_single_str_field(self):
        """Test collecting a uris: str field."""
        vector_cfg = MockVectorSourceConfig()
        result = collect_uris(vector_cfg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 'uris')
        self.assertEqual(result[0][1], '/path/to/data.geojson')

    def test_uris_multiple_items(self):
        """Test collecting a uris field with multiple URIs."""
        cfg = MockRasterSourceConfig()
        cfg.uris = ['/path/to/1.tif', '/path/to/2.tif', '/path/to/3.tif']
        result = collect_uris(cfg)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][0], 'uris[0]')
        self.assertEqual(result[1][0], 'uris[1]')
        self.assertEqual(result[2][0], 'uris[2]')

    def test_nested_model(self):
        """Test collecting URIs from a nested model structure."""
        scene = MockSceneConfig()
        scene.raster_source.uris = ['/path/to/scene.tif']
        scene.aoi_uris = ['/path/to/aoi.geojson']
        result = collect_uris(scene)
        paths = [r[0] for r in result]
        self.assertIn('raster_source.uris[0]', paths)
        self.assertIn('aoi_uris[0]', paths)

    def test_deeply_nested_model(self):
        """Test collecting URIs from a deeply nested model (scene > dataset)."""
        ds = MockDatasetConfig()
        ds.train_scenes[0].raster_source.uris = ['/path/to/scene.tif']
        ds.train_scenes[0].aoi_uris = ['/path/to/aoi.geojson']
        # Add a second scene
        scene2 = MockSceneConfig()
        scene2.id = 'scene2'
        scene2.raster_source.uris = ['/path/to/scene2.tif']
        ds.train_scenes.append(scene2)
        result = collect_uris(ds)
        paths = [r[0] for r in result]
        self.assertIn('train_scenes[0].raster_source.uris[0]', paths)
        self.assertIn('train_scenes[0].aoi_uris[0]', paths)
        self.assertIn('train_scenes[1].raster_source.uris[0]', paths)

    def test_pipeline_config(self):
        """Test collecting URIs from a PipelineConfig with dataset."""
        ds = MockDatasetConfig()
        ds.train_scenes[0].raster_source.uris = ['/path/to/train.tif']
        ds.train_scenes[0].aoi_uris = ['/path/to/train_aoi.geojson']

        pipeline_cfg = MockPipelineConfig()
        pipeline_cfg.dataset = ds
        pipeline_cfg.root_uri = '/tmp/output'

        result = collect_uris(pipeline_cfg)
        paths = [r[0] for r in result]
        self.assertIn('dataset.train_scenes[0].raster_source.uris[0]', paths)
        self.assertIn('dataset.train_scenes[0].aoi_uris[0]', paths)
        self.assertIn('root_uri', paths)

    def test_none_values_skipped(self):
        """Test that None values are skipped."""
        scene = MockSceneConfig()
        scene.aoi_uris = None  # explicitly None
        scene.vector_source = None
        scene.raster_source.uris = ['/path/to/img.tif']
        result = collect_uris(scene)
        paths = [r[0] for r in result]
        self.assertIn('raster_source.uris[0]', paths)
        self.assertNotIn('aoi_uris', paths)
        # vector_source should not cause errors even when None
        self.assertEqual(len(result), 1)

    def test_empty_list(self):
        """Test that empty lists return no URIs."""
        cfg = MockRasterSourceConfig()
        cfg.uris = []
        result = collect_uris(cfg)
        self.assertEqual(len(result), 0)

    def test_non_base_model_input(self):
        """Test that non-BaseModel inputs return empty list."""
        self.assertEqual(collect_uris('string'), [])
        self.assertEqual(collect_uris(42), [])
        self.assertEqual(collect_uris(None), [])

    def test_list_of_models(self):
        """Test collecting URIs from a list of models."""
        scenes = [
            MockSceneConfig(),
            MockSceneConfig(),
        ]
        scenes[0].raster_source.uris = ['/path/to/scene0.tif']
        scenes[1].raster_source.uris = ['/path/to/scene1.tif']
        result = collect_uris(scenes)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], '[0].raster_source.uris[0]')
        self.assertEqual(result[1][0], '[1].raster_source.uris[0]')

    def test_dict_value(self):
        """Test collecting URIs from a dict field."""
        cfg_dict = {'config': MockRasterSourceConfig(uris=['/path/to/img.tif'])}
        result = collect_uris(cfg_dict)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], '.config.uris[0]')


class TestCheckCommand(unittest.TestCase):
    """Tests for the rastervision check CLI command."""

    def setUp(self):
        self.runner = CliRunner()

    def test_check_help(self):
        """Test that --help works for the check command."""
        result = self.runner.invoke(main, ['check', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Validate all URIs', result.output)

    @patch('rastervision.pipeline.cli.file_exists')
    @patch('rastervision.pipeline.cli.get_configs')
    def test_check_with_missing_uri(self, mock_get_configs, mock_file_exists):
        """Test that missing URIs are reported."""
        # Mock a config with URIs
        pipeline_cfg = MockPipelineConfig()
        ds = MockDatasetConfig()
        ds.train_scenes[0].raster_source.uris = ['/tmp/missing.tif']
        pipeline_cfg.dataset = ds
        pipeline_cfg.root_uri = '/tmp/output'
        mock_get_configs.return_value = [pipeline_cfg]
        mock_file_exists.return_value = False

        result = self.runner.invoke(
            main,
            ['check', 'mock.module']
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn('missing', result.output)

    @patch('rastervision.pipeline.cli.file_exists')
    @patch('rastervision.pipeline.cli.get_configs')
    def test_check_all_found(self, mock_get_configs, mock_file_exists):
        """Test that found URIs are reported."""
        pipeline_cfg = MockPipelineConfig()
        ds = MockDatasetConfig()
        ds.train_scenes[0].raster_source.uris = ['/tmp/exists.tif']
        pipeline_cfg.dataset = ds
        pipeline_cfg.root_uri = '/tmp/output'
        mock_get_configs.return_value = [pipeline_cfg]
        mock_file_exists.return_value = True

        result = self.runner.invoke(
            main,
            ['check', 'mock.module']
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn('found', result.output)
        self.assertIn('✓', result.output)

    @patch('rastervision.pipeline.cli.file_exists')
    @patch('rastervision.pipeline.cli.get_configs')
    def test_check_with_args(self, mock_get_configs, mock_file_exists):
        """Test that -a args are passed to get_configs."""
        mock_get_configs.return_value = [MockPipelineConfig()]
        mock_file_exists.return_value = True

        result = self.runner.invoke(
            main,
            ['check', 'mock.module', '-a', 'root_uri', '/tmp/test']
        )
        self.assertEqual(result.exit_code, 0)
        mock_get_configs.assert_called_with(
            'mock.module', 'inprocess', {'root_uri': '/tmp/test'})

    @patch('rastervision.pipeline.cli.file_exists')
    @patch('rastervision.pipeline.cli.get_configs')
    def test_check_multiple_configs(self, mock_get_configs, mock_file_exists):
        """Test checking multiple PipelineConfigs."""
        cfg1 = MockPipelineConfig()
        cfg2 = MockPipelineConfig()
        mock_get_configs.return_value = [cfg1, cfg2]
        mock_file_exists.return_value = True

        result = self.runner.invoke(main, ['check', 'mock.module'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('PipelineConfig [0]', result.output)
        self.assertIn('PipelineConfig [1]', result.output)

    @patch('rastervision.pipeline.cli.file_exists')
    @patch('rastervision.pipeline.cli.get_configs')
    def test_check_no_uris(self, mock_get_configs, mock_file_exists):
        """Test checking a config with no URI fields."""
        cfg = MockPipelineConfig()  # No dataset set
        mock_get_configs.return_value = [cfg]

        result = self.runner.invoke(main, ['check', 'mock.module'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('No URIs found', result.output)

    @patch('rastervision.pipeline.cli.file_exists')
    @patch('rastervision.pipeline.cli.get_configs')
    def test_check_error_checking_uri(self, mock_get_configs, mock_file_exists):
        """Test error handling when file_exists throws."""
        pipeline_cfg = MockPipelineConfig()
        ds = MockDatasetConfig()
        ds.train_scenes[0].raster_source.uris = ['s3://bad-bucket/key.tif']
        pipeline_cfg.dataset = ds
        pipeline_cfg.root_uri = '/tmp/output'
        mock_get_configs.return_value = [pipeline_cfg]
        mock_file_exists.side_effect = Exception('Connection error')

        result = self.runner.invoke(main, ['check', 'mock.module'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('error checking', result.output)


if __name__ == '__main__':
    unittest.main()
