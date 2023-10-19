import os
import unittest

from rastervision.pipeline import rv_config_ as rv_config
from rastervision.aws_batch.aws_batch_runner import AWSBatchRunner


class MockPipeline:
    commands = ['test_cpu', 'test_gpu']
    split_commands = ['test_cpu']
    gpu_commands = ['test_gpu']


class TestAWSBatchRunner(unittest.TestCase):
    def test_build_cmd(self):
        pipeline = MockPipeline()
        runner = AWSBatchRunner()
        rv_config.set_verbosity(4)
        cmd, args = runner.build_cmd(
            'config.json',
            pipeline, ['predict'],
            num_splits=2,
            pipeline_run_name='test')
        cmd_expected = [
            'python', '-m', 'rastervision.pipeline.cli', '-vvv', 'run_command',
            'config.json', 'predict', '--runner', 'batch'
        ]
        args_expected = {
            'parent_job_ids': [],
            'num_array_jobs': None,
            'use_gpu': False,
            'job_queue': None,
            'job_def': None
        }
        self.assertListEqual(cmd, cmd_expected)
        self.assertTrue(args['job_name'].startswith('test'))
        del args['job_name']
        self.assertDictEqual(args, args_expected)

    def test_get_split_ind(self):
        runner = AWSBatchRunner()
        os.environ['AWS_BATCH_JOB_ARRAY_INDEX'] = '1'
        self.assertEqual(runner.get_split_ind(), 1)
        del os.environ['AWS_BATCH_JOB_ARRAY_INDEX']
        self.assertEqual(runner.get_split_ind(), 0)


if __name__ == '__main__':
    unittest.main()
