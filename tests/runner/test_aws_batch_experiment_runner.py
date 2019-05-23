import unittest
from unittest.mock import Mock

import rastervision as rv
from rastervision.rv_config import RVConfig
from rastervision.runner import AwsBatchExperimentRunner

import tests.mock as mk


class MockAwsBatchExperimentRunner(AwsBatchExperimentRunner):
    def __init__(self):
        super().__init__()

        self.mock_client = Mock()
        self.mock_client.submit_job.return_value = {'jobId': 'MOCK'}

    def _get_boto_client(self):
        return self.mock_client


class TestAwsBatchExperimentRunner(mk.MockMixin, unittest.TestCase):
    def test_respects_utilizes_gpu(self):
        config = self.mock_config()
        config['AWS_BATCH_job_queue'] = 'GPU_JOB_QUEUE'
        config['AWS_BATCH_job_definition'] = 'GPU_JOB_DEF'
        config['AWS_BATCH_cpu_job_queue'] = 'CPU_JOB_QUEUE'
        config['AWS_BATCH_cpu_job_definition'] = 'CPU_JOB_DEF'

        rv._registry.initialize_config(config_overrides=config)

        with RVConfig.get_tmp_dir() as tmp_dir:
            e = mk.create_mock_experiment().to_builder() \
                                           .with_root_uri(tmp_dir) \
                                           .clear_command_uris() \
                                           .build()

            runner = MockAwsBatchExperimentRunner()

            runner.run(
                e, commands_to_run=[rv.CHIP, rv.TRAIN, rv.PREDICT, rv.EVAL])

            submit_args = runner.mock_client.submit_job.call_args_list

            self.assertEqual(len(submit_args), 4)

            for args in submit_args:
                jobName, jobQueue = args[1]['jobName'], args[1]['jobQueue']

                if 'EVAL' in jobName or 'CHIP' in jobName:
                    self.assertTrue('CPU' in jobQueue)
                else:
                    self.assertTrue('GPU' in jobQueue)
