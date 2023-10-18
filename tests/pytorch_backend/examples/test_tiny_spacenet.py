import unittest
import shutil

from click.testing import CliRunner

from rastervision.pipeline.cli import main
from rastervision.pipeline.file_system.utils import get_tmp_dir
from rastervision.core.cli import predict

from tests import data_file_path


class TestTinySpacenet(unittest.TestCase):
    def test_rastervision_run_tiny_spacenet(self):
        runner = CliRunner()
        shutil.rmtree('/opt/data/output/tiny_spacenet', ignore_errors=True)
        result = runner.invoke(main, [
            'run', 'inprocess',
            'rastervision.pytorch_backend.examples.tiny_spacenet'
        ])
        self.assertEqual(result.exit_code, 0)

        # test predict command
        bundle_path = '/opt/data/output/tiny_spacenet/bundle/model-bundle.zip'
        img_path = data_file_path('small-rgb-tile.tif')
        with get_tmp_dir() as tmp_dir:
            result = runner.invoke(predict, [
                bundle_path, img_path, tmp_dir, '--channel-order', '0', '1',
                '2'
            ])
        self.assertEqual(result.exit_code, 0)


if __name__ == '__main__':
    unittest.main()
