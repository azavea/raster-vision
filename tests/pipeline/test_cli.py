import unittest
import shutil

from rastervision.pipeline.cli import main, print_error, convert_bool_args

from click.testing import CliRunner


class TestCli(unittest.TestCase):
    def test_rastervision_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        self.assertEqual(result.exit_code, 0)

    def test_rastervision_run_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ['run', '--help'])
        self.assertEqual(result.exit_code, 0)

    def test_rastervision_run_local(self):
        runner = CliRunner()
        shutil.rmtree('/opt/data/pipeline-example/1/', ignore_errors=True)
        result = runner.invoke(main, [
            'run', 'local', 'rastervision.pipeline_example_plugin1.config1',
            '-a', 'root_uri', '/opt/data/pipeline-example/1/', '--splits', '2'
        ])
        self.assertEqual(result.exit_code, 0)

        # from config path
        cfg_path = ('rastervision_pipeline/rastervision/'
                    'pipeline_example_plugin1/config1.py')
        result = runner.invoke(main, [
            'run', 'local', cfg_path, '-a', 'root_uri',
            '/opt/data/pipeline-example/1/', '--splits', '2'
        ])
        self.assertEqual(result.exit_code, 0)

    def test_rastervision_run_inprocess1(self):
        runner = CliRunner()
        shutil.rmtree('/opt/data/pipeline-example/1/', ignore_errors=True)
        result = runner.invoke(main, [
            'run', 'inprocess',
            'rastervision.pipeline_example_plugin1.config1', '-a', 'root_uri',
            '/opt/data/pipeline-example/1/', '--splits', '2'
        ])
        self.assertEqual(result.exit_code, 0)

    def test_rastervision_run_inprocess2(self):
        runner = CliRunner()
        shutil.rmtree('/opt/data/pipeline-example/2/', ignore_errors=True)
        result = runner.invoke(main, [
            'run', 'inprocess',
            'rastervision.pipeline_example_plugin1.config2', '-a', 'root_uri',
            '/opt/data/pipeline-example/2/', '--splits', '2'
        ])
        self.assertEqual(result.exit_code, 0)

    def test_rastervision_run_inprocess3(self):
        runner = CliRunner()
        shutil.rmtree('/opt/data/pipeline-example/3/', ignore_errors=True)
        result = runner.invoke(main, [
            'run', 'inprocess',
            'rastervision.pipeline_example_plugin2.config3', '-a', 'root_uri',
            '/opt/data/pipeline-example/3/', '--splits', '2'
        ])
        self.assertEqual(result.exit_code, 0)


class TestUtils(unittest.TestCase):
    def test_print_error(self):
        print_error('error')

    def test_convert_bool_args(self):
        args_in = dict(a='true', b='false')
        args_out = convert_bool_args(args_in)
        self.assertDictEqual(args_out, dict(a=True, b=False))


if __name__ == '__main__':
    unittest.main()
