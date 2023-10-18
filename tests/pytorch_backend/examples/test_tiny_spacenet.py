import unittest
import shutil

from click.testing import CliRunner

from rastervision.pipeline.cli import main


class TestTinySpacenet(unittest.TestCase):
    def test_rastervision_run_tiny_spacenet(self):
        runner = CliRunner()
        shutil.rmtree('/opt/data/output/tiny_spacenet', ignore_errors=True)
        result = runner.invoke(main, [
            'run', 'inprocess',
            'rastervision.pytorch_backend.examples.tiny_spacenet'
        ])
        self.assertEqual(result.exit_code, 0)


if __name__ == '__main__':
    unittest.main()
