import unittest
from os.path import join
import shutil

from click.testing import CliRunner

from rastervision.pipeline.cli import main
from rastervision.pipeline.file_system.utils import get_tmp_dir, json_to_file
from rastervision.core.cli import predict, predict_scene
from rastervision.core.data import (RasterioSourceConfig, SceneConfig,
                                    SemanticSegmentationLabelStoreConfig)
from rastervision.core.rv_pipeline import SemanticSegmentationPredictOptions

from tests import data_file_path


class TestTinySpacenet(unittest.TestCase):
    def test_rastervision_run_tiny_spacenet(self):
        runner = CliRunner()
        shutil.rmtree('/opt/data/output/tiny_spacenet', ignore_errors=True)
        result = runner.invoke(main, [
            'run', 'inprocess',
            'rastervision.pytorch_backend.examples.tiny_spacenet'
        ])
        if result.exit_code != 0:
            raise result.exception

        # test predict command
        bundle_path = '/opt/data/output/tiny_spacenet/bundle/model-bundle.zip'
        img_path = data_file_path('small-rgb-tile.tif')
        with get_tmp_dir() as tmp_dir:
            result = runner.invoke(predict, [
                bundle_path, img_path, tmp_dir, '--channel-order', '0', '1',
                '2'
            ])
        if result.exit_code != 0:
            raise result.exception

        # test predict_scene command
        bundle_path = '/opt/data/output/tiny_spacenet/bundle/model-bundle.zip'
        img_path = data_file_path('small-rgb-tile.tif')
        with get_tmp_dir() as tmp_dir:
            pred_uri = join(tmp_dir, 'pred')
            rs_cfg = RasterioSourceConfig(uris=img_path)
            ls_cfg = SemanticSegmentationLabelStoreConfig(uri=pred_uri)
            scene_cfg = SceneConfig(
                id='', raster_source=rs_cfg, label_store=ls_cfg)
            pred_opts = SemanticSegmentationPredictOptions()
            scene_config_uri = join(pred_uri, 'scene-config.json')
            json_to_file(scene_cfg.dict(), scene_config_uri)
            pred_opts_uri = join(pred_uri, 'predict-options.json')
            json_to_file(pred_opts.dict(), pred_opts_uri)
            result = runner.invoke(predict_scene, [
                bundle_path, scene_config_uri, '--predict_options_uri',
                pred_opts_uri
            ])
        if result.exit_code != 0:
            raise result.exception


if __name__ == '__main__':
    unittest.main()
